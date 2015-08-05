#
# LucidDream - A deepdream program with copious options
# Using a fork of Bat Country (https://github.com/jrosebr1/bat-country)
#

from batcountry import BatCountry
from PIL import Image
import PIL.ImageOps
import numpy as np
import argparse
import scipy.ndimage as nd
import re
import math

#  -----------------
#	Argument Parser
#  -----------------

ap = argparse.ArgumentParser()

ap.add_argument("-b", "--base-model", type=str, default="google", help="base model path")
ap.add_argument("-l", "--layer", type=str, default="conv2/3x3", help="layer of CNN to use.")

ap.add_argument("-i", "--image", required=True, help="path to base image")
ap.add_argument("-o", "--output", required=True, help="path to output image")

ap.add_argument("-v", "--vis", type=str, help="path to output directory for visualizations")

ap.add_argument("-oc", "--octaves", type=int, default=4, help="number of octaves")
ap.add_argument("-ol", "--octave-layer", type=str, default=None, help="list of comma-separated layers for each octave")
ap.add_argument("-oi", "--octave-iterations", type=str, default=None, help="iterations for each octave, as a list or a range")
ap.add_argument("-os", "--octave-scale", type=float, default=1.4, help="scale for each octave, as a value, a list or a range")

ap.add_argument("-it", "--iterations", type=str, default="5", help="number of iterations")

ap.add_argument("-gr", "--gradient", type=str, default=None, help="center for zoom")
ap.add_argument("-gc", "--gradient-center", type=str, default=None, help="center for zoom")

ap.add_argument("-d", "--dreams", type=int, default=1, help="number of dreams")

ap.add_argument("-z", "--zoom", type=float, default=None, help="scale to zoom to (default no zoom)")
ap.add_argument("-c", "--center", type=str, default=None, help="center for zoom")
ap.add_argument("-r", "--rotation", type=float, default=0.0, help="rotation for each zoom to (default 0 deg)")

#in ap.add_argument("-in", "--interpolate", type=int, default=None, help="import existing dream")

ap.add_argument("-g", "--guide-image", type=str, default=None, help="path to guide image")

#in ap.add_argument("-im", "--import-dream", type=str, default=None, help="import existing dream")

args = ap.parse_args()


# -----------------
# Library Functions
# -----------------


def scale_dif(image,full_size):
	#compare images
	x,y = image.shape[0:2]
	i,j = full_size.shape[0:2]
	print("scale dif {} {} {} {}".format(x,y,i,j))
	if i == x and j == y:
		return image
	original = nd.zoom(full_size, (1.0*x/i,1.0*y/j,1))
	dif_zoom = nd.zoom(original - image, (1.0*i/x,1.0*j/y,1))
	
	return np.clip(full_size - dif_zoom,0,255)

# Transform image
def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None,expand=False):
	if center is None:
		return image.rotate(angle)
	angle = -angle/180.0*math.pi
	nx,ny = x,y = center
	sx=sy=1.0
	if new_center:
		(nx,ny) = new_center
	if scale:
		(sx,sy) = scale
	cosine = math.cos(angle)
	sine = math.sin(angle)
	a = cosine/sx
	b = sine/sx
	c = x-nx*a-ny*b
	d = -sine/sy
	e = cosine/sy
	f = y-nx*d-ny*e
	return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.BICUBIC)	

# Feather edges of a gradient mask
def feather_gradient(mask,ratio=0.1):
	x,y = mask.size[:2]
	feather_x = int(x*ratio)
	feather_y = int(y*ratio)
	for i in range(feather_x):
		feather_val = int(255*i/feather_x)
		for j in range(y):
			if mask.getpixel((i,j)) > feather_val:
				mask.putpixel((i,j),feather_val)
			if mask.getpixel((x-i-1,j)) > feather_val:
				mask.putpixel((x-i-1,j),feather_val)
	for i in range(feather_y):
		feather_val = int(255*i/feather_y)
		for j in range(x):
			if mask.getpixel((j,i)) > feather_val:
				mask.putpixel((j,i),feather_val)
			if mask.getpixel((j,y-i-1)) > feather_val:
				mask.putpixel((j,y-i-1),feather_val)
	return mask

# Create radial gradient mask
def radial_gradient(width,height,center_x,center_y,r_inner,r_outer):
	#create a radial gradient mask
	mask = Image.new('L', (width,height), 0)
	r_range = r_outer - r_inner
	for x in range(width):
		for y in range(height):
			#for each pixel, get distance from center
			r = math.sqrt((x-center_x)*(x-center_x) + (y-center_y)*(y-center_y))
			if r < r_inner:
				mask.putpixel((x,y),255)
			elif r < r_outer:
				mask.putpixel((x,y),255*(r_outer - r)/r_range)
	return mask

# For an image of size w,h, and a circle center x,y
# at what radius will the circle fill the image?
def max_radius(w,h,x,y):
	return max( ( 
		math.sqrt(x*x + y*y) ,
		math.sqrt((w-x)*(w-x) + (h-y)*(h-y)) ,
		math.sqrt(x*x + (h-y)*(h-y)) ,
		math.sqrt((w-x)*(w-x) + y*y) ) )
	

# create a set of radial gradients	
def gradient_set(width,height,center_x,center_y,num):
	#create a set of radial masks that go from 0 to 255, spreading out from center
	#first get the maximum radius required
	max_r = max_radius(width,height,center_x,center_y)
	#print("max distance: {} center: {},{} in {},{}".format(max_r,center_x,center_y,width,height))
	
	#now go through your number
	#in the first half send out the outer layer
	#in the second half send out the inner layer
	half_num = int(num/2)
	mask_set = []
	for i in range(half_num):
		mask_set.append( radial_gradient(width,height,center_x,center_y,0,max_r*i/(half_num-1)) )
	for i in range(num - half_num):
		mask_set.append( radial_gradient(width,height,center_x,center_y,max_r*i/(num - half_num-1),max_r) )
	
	return mask_set
		

# deepdream an image
def dream(base_img,base_name,layer,iter,guide_image = None,obj_fn = None, obj_feat = None):
	if args.vis:
		if obj_feat is None:
			(image, visualizations) = bc.dream(base_img, iter_n=iter, 
				octave_scale=dream_o_scale, octave_n=dream_o,  end=layer, visualize=True)
		else:
			(image, visualizations) = bc.dream(base_img, iter_n=iter,
				objective_fn=obj_fn, objective_features=obj_feat, 
				octave_scale=dream_o_scale, octave_n=dream_o,  end=layer, visualize=True)

		# loop over the visualizations
		count = 0
		vis_list = []
		for (k, vis) in visualizations:
			# write the visualization to file
			outputPath = "{}_vis_{}.jpg".format(base_name,count)
			count = count + 1
			result = scale_dif(vis,base_img)
			result = Image.fromarray(np.uint8(result))
			result.save(outputPath)
			vis_list.append(result)
		
		if args.gradient is not None:
			image_start = Image.fromarray(np.uint8(base_img))
			image_x, image_y = image_start.size[0], image_start.size[1]
			image_range = len(vis_list)
			center_x = image_x/2
			center_y = image_y/2
			set_of_gradients = gradient_set(image_x,image_y,int(center_x),int(center_y),gradient_steps)		
			sandbox = Image.new('RGB', (image_x,image_y))
			total_img = (np.sum(iter) - 1) * image_steps + gradient_steps
			for i in range(total_img):
				#print("sandbox {} {} {} {}".format(image_x,image_y,image_range,total_img))
				sandbox.paste(image_start,(0,0,image_x,image_y))
				for j in range(image_range):
					#calculate current value, based on i
					#this one should turn up at j*image_steps, before that, don't paste
					if i >= j*image_steps:
						if i >= j*image_steps+gradient_steps:
							#if it's after j*image_steps + gradient_steps then just paste
							sandbox.paste(vis_list[j],(0,0,image_x,image_y))
						else:
							#choose correct gradient
							grad = i-j*image_steps
							#print("correct gradient is {} {} {}".format(vis_list[j].size[0],vis_list[j].size[1],grad))
							sandbox.paste(vis_list[j],(0,0,image_x,image_y),set_of_gradients[grad])
				sandbox.save('{}_grad_{}.jpg'.format(base_name,i))
				
		result = Image.fromarray(np.uint8(image))
		result.save("{}.jpg".format(base_name))
		
		return image
	else:
		print("dream {}".format(base_name))
		if obj_feat is None:
			image = bc.dream(base_img, end=layer,iter_n=iter, octave_n=dream_o,
				octave_scale=dream_o_scale)
		else:
			image = bc.dream(base_img, end=layer,iter_n=iter, octave_n=dream_o,
				octave_scale=dream_o_scale, objective_fn=obj_fn, objective_features=obj_feat)
		result = Image.fromarray(np.uint8(image))
		result.save("{}.jpg".format(base_name))
		
		return image

def zoom_dream(img,zoom_num,layer,iter,base_name,guide_image = None):

	iter_args = map(int,get_args_list(args.octave_iterations,dream_o,iter[0]))
	
	if guide_image is not None:
		if type(layer) is not list:
			features = bc.prepare_guide(Image.open(guide_image), end=layer)
		else:
			features = bc.prepare_guide(Image.open(guide_image), end=layer[0])		
		obj_fn = BatCountry.guided_objective
	else:
		features = None
		obj_fn = None
	
	orig_dream = dream(np.float32(img), base_name + "_0" ,layer,iter_args,obj_fn = obj_fn,obj_feat = features)
	print("dream 1/{} complete".format(zoom_num+1))
	
	zoom = [Image.fromarray(np.uint8(orig_dream))]
	for a in range(zoom_num):
		if args.zoom is None:
			img_zoom = zoom[-1]
		else:
			zoom_center_x = img_zoom_x + ( base_center_dif_x*(1.0*a/zoom_num ))
			zoom_center_next_x = img_zoom_x + ( base_center_dif_x*(1.0*(a+1.0)/zoom_num) )
			zoom_center_y = img_zoom_y + ( base_center_dif_y*(1.0*a/zoom_num ))
			zoom_center_next_y = img_zoom_y + ( base_center_dif_y*(1.0*(a+1.0)/zoom_num) )
			iter_args = map(int,get_args_list(args.octave_iterations,dream_o,iter[a]))
			
			
			img_zoom = ScaleRotateTranslate(zoom[-1], zoom_a, 
				center = (zoom_center_x,zoom_center_y), new_center = (zoom_center_next_x,zoom_center_next_y), 
				scale = (1.0+zoom_s,1.0+zoom_s))
			print("centers: half {},{} zoom {},{} center_dif {},{}".format(base_half_x,base_half_y,img_zoom_x,img_zoom_y,base_center_dif_x,base_center_dif_y))
			print("zooming in: {}% rotating: {} degrees center: {},{} -> {},{}".format(int(zoom_s*100),zoom_a,zoom_center_x,zoom_a,zoom_center_y,zoom_center_next_x,zoom_center_next_y))
		dream_array = dream( np.float32(img_zoom), base_name + "_" + str(a+1), layer, iter_args,obj_fn = obj_fn,obj_feat = features)
		print("dream {}/{} complete".format(a+2,zoom_num+1))
		img_dream = Image.fromarray(np.uint8(dream_array))
		
		zoom.append( img_dream )
	return zoom

def get_args_list(data,length,init=None):
	if data is None:
		data_list = [init] * length
	elif re.search("^([0-9]+|[0-9]*\.[0-9]*)-([0-9]+|[0-9]*\.[0-9]*)$",data):
		a,b = [int(n) for n in data.split('-')]
		data_list = []
		for i in range(length):
			data_list.append( a + 1.0*i*(b-a)/(length-1) )
	else:
		data_list = data.split(',')
		if len(data_list) > 1:
			if data_list[0] == "" and init is not None:
				data_list[0] = init
			for i in range(1,len(data_list)):
				if data_list[i] == "":
					data_list[i] = data_list[i-1]
			for i in range(len(data_list),length):
				data_list.append(data_list[i-1])
		else:
			data_list = [data] * length
	return data_list

		



input_base = args.image
dreams = args.dreams

dream_o = args.octaves
dream_o_scale = args.octave_scale
dream_i = map(int,get_args_list(args.iterations,dreams))

zoom_num = dreams - 1
zoom_s = args.zoom
zoom_a = args.rotation




base_img = Image.open(input_base)
base_x, base_y = base_img.size[0:2]
base_mask = Image.new('L', (base_x,base_y), 255)




#for gradient visualisation, get two variables:
# gradient_steps: how long it takes a visualisation frame to show
# image_steps: the gap between each visualisation frame

if args.gradient:
	vis_data = args.graident.split(',')
	gradient_steps = int(vis_data[0])
	if len(vis_data) > 1:
		image_steps = int(vis_data[1])
	else:
		image_steps = 0

	total_img = (np.sum(dream_i) - 1) * image_steps + gradient_steps
else:
	if args.vis:
		total_img = np.sum(dream_i)
	else:
		total_img = 0


# Get centres for zoom and gradient visualisation.

base_half_x = base_x/2
base_half_y = base_y/2
if args.center is None:
	base_center_x, base_center_y = base_half_x, base_half_y
else:
	base_center_x, base_center_y = map(float,args.center.split(','))

base_center_dif_x = base_half_x - base_center_x
base_center_dif_y = base_half_y - base_center_y

if args.gradient_center is None:
	gradient_center_x, gradient_center_y = base_center_x, base_center_y
else:
	gradient_center_x, gradient_center_y = map(float,args.gradient_center.split(','))




google_layers= 'conv1/7x7_s2,conv2/3x3,conv2/3x3_reduce,conv2/norm2,inception_3a/1x1,inception_3a/3x3,inception_3a/3x3_reduce,inception_3a/5x5,inception_3a/5x5_reduce,inception_3a/output,inception_3a/pool,inception_3a/pool_proj,inception_3b/1x1,inception_3b/3x3,inception_3b/3x3_reduce,inception_3b/5x5,inception_3b/5x5_reduce,inception_3b/output,inception_3b/pool,inception_3b/pool_proj,inception_4a/1x1,inception_4a/3x3,inception_4a/3x3_reduce,inception_4a/5x5,inception_4a/5x5_reduce,inception_4a/output,inception_4a/pool,inception_4a/pool_proj,inception_4b/1x1,inception_4b/3x3,inception_4b/3x3_reduce,inception_4b/5x5,inception_4b/5x5_reduce,inception_4b/output,inception_4b/pool,inception_4b/pool_proj,inception_4c/1x1,inception_4c/3x3,inception_4c/3x3_reduce,inception_4c/5x5,inception_4c/5x5_reduce,inception_4c/output,inception_4c/pool,inception_4c/pool_proj,inception_4d/1x1,inception_4d/3x3,inception_4d/3x3_reduce,inception_4d/5x5,inception_4d/5x5_reduce,inception_4d/output,inception_4d/pool,inception_4d/pool_proj,inception_4e/1x1,inception_4e/3x3,inception_4e/3x3_reduce,inception_4e/5x5,inception_4e/5x5_reduce,inception_4e/output,inception_4e/pool,inception_4e/pool_proj,inception_5a/1x1,inception_5a/3x3,inception_5a/3x3_reduce,inception_5a/5x5,inception_5a/5x5_reduce,inception_5a/output,inception_5a/pool,inception_5a/pool_proj,inception_5b/1x1,inception_5b/3x3,inception_5b/3x3_reduce,inception_5b/5x5,inception_5b/5x5_reduce,inception_5b/output,inception_5b/pool,inception_5b/pool_proj,pool1/3x3_s2,pool1/norm1,pool2/3x3_s2,pool3/3x3_s2,pool4/3x3_s2'
place_layers= 'conv1/7x7_s2,pool1/3x3_s2,pool1/norm1,conv2/3x3_reduce,conv2/3x3,conv2/norm2,pool2/3x3_s2,inception_3a/1x1,inception_3a/3x3_reduce,inception_3a/3x3,inception_3a/5x5_reduce,inception_3a/5x5,inception_3a/pool,inception_3a/pool_proj,inception_3a/output,inception_3b/1x1,inception_3b/3x3_reduce,inception_3b/3x3,inception_3b/5x5_reduce,inception_3b/5x5,inception_3b/pool,inception_3b/pool_proj,inception_3b/output,pool3/3x3_s2,inception_4a/1x1,inception_4a/3x3_reduce,inception_4a/3x3,inception_4a/5x5_reduce,inception_4a/5x5,inception_4a/pool,inception_4a/pool_proj,inception_4a/output,inception_4b/1x1,inception_4b/3x3_reduce,inception_4b/3x3,inception_4b/5x5_reduce,inception_4b/5x5,inception_4b/pool,inception_4b/pool_proj,inception_4b/output,inception_4c/1x1,inception_4c/3x3_reduce,inception_4c/3x3,inception_4c/5x5_reduce,inception_4c/5x5,inception_4c/pool,inception_4c/pool_proj,inception_4c/output,inception_4d/1x1,inception_4d/3x3_reduce,inception_4d/3x3,inception_4d/5x5_reduce,inception_4d/5x5,inception_4d/pool,inception_4d/pool_proj,inception_4d/output,inception_4e/1x1,inception_4e/3x3_reduce,inception_4e/3x3,inception_4e/5x5_reduce,inception_4e/5x5,inception_4e/pool,inception_4e/pool_proj,inception_4e/output,pool4/3x3_s2,inception_5a/1x1,inception_5a/3x3_reduce,inception_5a/3x3,inception_5a/5x5_reduce,inception_5a/5x5,inception_5a/pool,inception_5a/pool_proj,inception_5a/output,inception_5b/1x1,inception_5b/3x3_reduce,inception_5b/3x3,inception_5b/5x5_reduce,inception_5b/5x5,inception_5b/pool,inception_5b/pool_proj,inception_5b/output'
car_layers= 'conv1,pool1,norm1,conv2_1x1,conv2_3x3,norm2,pool2,inception_3a_1x1,inception_3a_3x3_reduce,inception_3a_3x3,inception_3a_5x5_reduce,inception_3a_5x5,inception_3a_pool,inception_3a_pool_proj,inception_3a_output,inception_3b_1x1,inception_3b_3x3_reduce,inception_3b_3x3,inception_3b_5x5_reduce,inception_3b_5x5,inception_3b_pool,inception_3b_pool_proj,inception_3b_output,pool3,inception_4a_1x1,inception_4a_3x3_reduce,inception_4a_3x3,inception_4a_5x5_reduce,inception_4a_5x5,inception_4a_pool,inception_4a_pool_proj,inception_4a_output,inception_4b_1x1,inception_4b_3x3_reduce,inception_4b_3x3,inception_4b_5x5_reduce,inception_4b_5x5,inception_4b_pool,inception_4b_pool_proj,inception_4b_output,inception_4c_1x1,inception_4c_3x3_reduce,inception_4c_3x3,inception_4c_5x5_reduce,inception_4c_5x5,inception_4c_pool,inception_4c_pool_proj,inception_4c_output,inception_4d_1x1,inception_4d_3x3_reduce,inception_4d_3x3,inception_4d_5x5_reduce,inception_4d_5x5,inception_4d_pool,inception_4d_pool_proj,inception_4d_output,inception_4e_1x1,inception_4e_3x3_reduce,inception_4e_3x3,inception_4e_5x5_reduce,inception_4e_5x5,inception_4e_pool,inception_4e_pool_proj,inception_4e_output,pool4,inception_5a_1x1,inception_5a_3x3_reduce,inception_5a_3x3,inception_5a_5x5_reduce,inception_5a_5x5,inception_5a_pool,inception_5a_pool_proj,inception_5a_output,inception_5b_1x1,inception_5b_3x3_reduce,inception_5b_3x3,inception_5b_5x5_reduce,inception_5b_5x5,inception_5b_pool,inception_5b_pool_proj,inception_5b_output'


output_root = re.sub("\.jpg$","",args.output)
model_list = args.base_model.split(',')
if args.guide_image is not None:
	guide_list = args.guide_image.split(',')

for base_model in model_list:
	# CNN model
	if len(model_list) > 1:
		output_base = output_root + "_" + base_model
	else:
		output_base = output_root
	if base_model == "place":
		all_layers = place_layers.split(',')
	elif base_model == "cars":
		all_layers = car_layers.split(',')
	else:
		all_layers = google_layers.split(',')

	# CNN layers
	layer_set = args.layer.split(',')
	layer_list = []
	for i in layer_set:
		if i == 'all':
			layer_list = all_layers
		else:
			if re.search( r'/all$',i,re.M|re.I):
				layer_root = re.sub("\/","\/",re.sub("/all$","/*",i))
				for j in all_layers:
					if re.search(layer_root,j):
						layer_list.append(j)
			else:
				layer_list.append(i)

	#quick fix for the car deploy.txt - for some reason it needs to be _ instead of /
	if base_model == "cars":
		def car_re(l): return re.sub("\/","_",l)
		layer_list = map(car_re,layer_list)
	
	bc = BatCountry(base_model)
	if args.guide_image is None:
		if len(layer_list) > 1:
			for layer in layer_list:
				layer_args = get_args_list(args.octave_layer,dream_o,layer)
				base_name = output_base+'_'+re.sub("\/","_",layer)
				zdream = zoom_dream(base_img,zoom_num,layer_args,dream_i,base_name,None)
		else:
			layer_args = get_args_list(args.octave_layer,dream_o,layer_list[0])
			zdream = zoom_dream(base_img,zoom_num,layer_args,dream_i,output_base,None)
	else:
		for guide_image in guide_list:
			if len(guide_list) > 1:
				output_stub = output_base + "_" + re.sub("[^a-zA-Z_0-9]","_",guide_image)
			else:
				output_stub = output_base
			if len(layer_list) > 1:
				for layer in layer_list:
					layer_args = get_args_list(args.octave_layer,dream_o,layer)
					base_name = output_stub+'_'+re.sub("\/","_",layer)
					zdream = zoom_dream(base_img,zoom_num,layer_args,dream_i,base_name,guide_image)
			else:
				layer_args = get_args_list(args.octave_layer,dream_o,layer_list[0])
				zdream = zoom_dream(base_img,zoom_num,layer_args,dream_i,output_stub,guide_image)
			
	bc.cleanup()
