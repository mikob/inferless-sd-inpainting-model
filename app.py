from supabase import create_client, Client
from PIL import Image
import torch
import time
import os
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline

supabase_conn: Client = create_client(os.environ['PUBLIC_SUPABASE_URL'], os.environ['SUPABASE_API_KEY'])
bucket = supabase_conn.storage.from_('studio')
SKIP_RENDERING = os.environ.get('SKIP_RENDERING', 'false').lower() == 'true'

lengths = ["short", "medium length", "long"]
default_hair_color = 'dark brown'
def get_prompts(male: bool, density: int, hair_color: str, widows_peak: bool) -> Tuple[str, str]:
	# if l == 'buzz cut':
	# 	yield f"attractive young man, {l}"
	# 	yield f"male model, {l}"
	# yield f"attractive young woman, {l}, {hair_color} hair"
	# yield f"attractive young african man, buzzed neat black hair, symmetrical, neat hairline, angled top of head"
	if hair_color == 'automatic':
		hair_w_color = f'{default_hair_color} hair'
	else:
		hair_w_color = f'{hair_color} hair'

	if density == 50:
		return [f'photo of a {"man" if male else "woman" }, messy short {hair_w_color} grafts{" widows peak" if widows_peak else ""}',
			"rendering, 3d, deformed, mutated, ugly, hat, headband, cap, blurry, glasses, sunglasses"]
	else:
		return [f'photo of an attractive young {"man" if male else "woman" }, messy {lengths[0]} {hair_w_color} {" widows peak" if widows_peak else ""}',
			"rendering, 3d, receding, temples, deformed, mutated, ugly, bald, balding, big-forehead, middle-aged, forehead, hat, old, headband, cap, blurry, glasses, sunglasses"]
	# yield `male model, ${l}, ${hair_color} hair`;


def dummy_safety_checker(images, **kwargs): 
	return images, False 


class InferlessPythonModel:
	def initialize(self):
		self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
			"runwayml/stable-diffusion-inpainting",
			torch_dtype=torch.float16
		)
		self.pipe.to("cuda:0")
		self.pipe.safety_checker = dummy_safety_checker

	def infer(self, inputs):
		email = inputs['email']
		# aka gallery password
		session = inputs['session']
		# for embeds - they always upload to a unique location to prevent overwriting 
		# photos from the same user
		upload_id = inputs.get('upload_id', '')
		gallery_id = inputs['gallery_id']
		photo_id = inputs['photo_id']
		mask_id = inputs['mask_id']
		iteration_id = inputs['iteration_id']
		count = inputs.get('count', 4)
		size = inputs.get('size', 1024)
		embed = inputs.get('embed', False)
		try:
			progress_0_data, progress_0_count = supabase_conn.table("trackers").update({
				"progress": 0,
			}).eq("gallery_id", gallery_id).eq("photo_id", photo_id).eq("mask_id", mask_id).eq("iteration_id", iteration_id).execute()
		except e:
			print(f'Error updating tracker to progress 0! Continuing... {e}')

		if len(progress_0_data[1]) == 0:
			raise Exception(f"No tracker found for gallery_id {gallery_id}, photo_id {photo_id}, mask_id {mask_id}, iteration_id {iteration_id}")
		tracker = progress_0_data[1][0]
		_pos_prompt, _neg_prompt = get_prompts(tracker['male'], tracker['density'], tracker['hair_color'], tracker['widows_peak'])
		prompt = inputs.get('prompt', _pos_prompt)
		negative_prompt = inputs.get('negative_prompt', _neg_prompt)

		start_time = time.time()

		# download the photo from supabase
		photo_stream = bucket.download(f'{email}/{session}/{photo_id}_proc.png')
		# download the mask from supabase
		mask_stream = bucket.download(f'{email}/{session}/{photo_id}_{mask_id}.png')
		print(f'Downloading photo and mask took {time.time() - start_time} seconds')

		gen_start_time = time.time()
		photo = Image.open(BytesIO(photo_stream))
		mask = Image.open(BytesIO(mask_stream))

		if SKIP_RENDERING:
			gend_images = [photo]
		else:
			gend_images = self.pipe(
				width=size,
				height=size,    
				prompt=prompt, 
				negative_prompt=negative_prompt,
				count=count,
				image=photo, mask_image=mask, num_inference_steps=35, guidance_scale=7.5
			).images
		print(f'Generating image(s) took {time.time() - gen_start_time} seconds')

		upload_start_time = time.time()
		for _idx, gend_image in enumerate(gend_images):
			img_byte_arr = BytesIO()
			gend_image.save(img_byte_arr, format='WEBP')
			# overwrite (in case we're doing a regen)
			# TODO: catch exceptions on this upload -seems to have failed with athenshair before
			bucket.upload(f'{email}/{session}/{photo_id}_{mask_id}_{iteration_id}.webp', img_byte_arr.getvalue(), { "content-type": "image/webp", "x-upsert": "true" })
			try:
				progress_100_data, progress_100_count = supabase_conn.table("trackers").update({
					"progress": 100,
				}).eq("gallery_id", gallery_id).eq("photo_id", photo_id).eq("mask_id", mask_id).eq("iteration_id", iteration_id).execute()
			except e:
				print(f'Error updating tracker (gallery_id: {gallery_id}, photo_id: {photo_id}, mask_id: {mask_id}, iteration_id: {iteration_id}) to progress 100! Continuing... {e}')

		lossless_start_time = time.time()
		print(f'Converting to webp and uploading image(s) to supabase took {lossless_start_time - upload_start_time} seconds')

		# generated via the widget
		if embed:
			# always a photo_id of 0 because backend keeps track of the photo_id
			bucket.copy(f'{email}/{session}/{photo_id}_{mask_id}_{iteration_id}.webp', f'embed@hairgen.ai/{upload_id}/0_{mask_id}_{iteration_id}.webp')
		else:
			# now upload the originals to supabase (for the gallery dl button)
			for iteration_id, gend_image in enumerate(gend_images):
				img_byte_arr = BytesIO()
				gend_image.save(img_byte_arr, format='png')
				# overwrite (in case we're doing a regen)
				# TODO: catch exceptions on this upload -seems to have failed with athenshair before
				bucket.upload(f'{email}/{session}/{photo_id}_{mask_id}_{iteration_id}.png', img_byte_arr.getvalue(), { "content-type": "image/png", "x-upsert": "true" })

		print(f'Converting to lossless png and uploading image(s) to supabase took {time.time() - lossless_start_time} seconds')
		return True

	def finalize(self):
		self.pipe = None
