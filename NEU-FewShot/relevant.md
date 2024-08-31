## 提纲

题目：Improving Few-Shot Steel Surface Defect Detection: An Automated Data Augmentation Framework Using Large-Scale Foundation Models

ref: 

[1] [Few-Shot Steel Surface Defect Detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9623623)

**1.Introduction**

**2.Related works**

2.1 Few-shot learning 

2.2  Large-Scale Foundation Models

**3.Methodology**

3.1Prompt engineering

- label to text（English）： I need a search words for image generation. The image is about "steed......"

	input:  I need a prompt for stable diffusion image generation. The image is about the {label} defect on rolled steel surface.  Could you write a prompt for me?

	|defect type|prompt|
	| :---: |:---------|
	|crazing|Create a high-resolution image depicting the crazing defect on a rolled steel surface. The steel surface should appear smooth and metallic, with visible lines and patterns characteristic of rolled steel. Highlight the crazing defect with fine, irregular cracks that spread across the surface, forming a network of fissures. The cracks should vary in length and width, creating a distinct contrast against the otherwise uniform texture of the steel. The lighting should emphasize the texture and defects, casting subtle shadows within the cracks to enhance their visibility. The overall mood should be industrial, focusing on the detail and texture of the steel and its imperfections.|
	|inclusion|Create a high-resolution image depicting the inclusion defect on a rolled steel surface. The steel surface should appear smooth and metallic, with characteristic lines and patterns from the rolling process. Highlight the inclusion defect as irregular, embedded particles or impurities within the steel, creating visible dark spots or streaks that disrupt the uniform texture. The inclusions should vary in size and shape, appearing as small clusters or singular anomalies. The lighting should emphasize the texture and defects, casting subtle shadows around the inclusions to enhance their visibility. The overall mood should be industrial, focusing on the detail and texture of the steel and its imperfections.|
	|patches|A close-up view of a rolled steel surface showing various patches defects. The steel is industrial-grade, with a metallic gray appearance. The defects are irregularly shaped patches, some slightly raised and others indented, disrupting the otherwise smooth surface. The lighting highlights the texture and imperfections, casting subtle shadows around the defects. The background is blurred to keep the focus on the steel surface, emphasizing the contrast between the smooth and defective areas. The image should convey an industrial setting, with a realistic and detailed depiction of the defects.|
	|pitted_surface|A close-up view of a rolled steel surface exhibiting pitted surface defects. The steel has a metallic gray finish with a slightly reflective quality. The defects appear as small, irregularly shaped pits and depressions scattered across the surface, creating a rough texture. The lighting is angled to accentuate the depth and details of the pits, casting small shadows and highlighting the variations in the surface. The background is softly blurred to keep the focus on the steel surface, emphasizing the contrast between the smooth areas and the pitted defects. The image should convey a realistic industrial setting, showcasing the imperfections in detail.|
	|rolled-in_scale|A close-up view of a rolled steel surface showing rolled-in scale defects. The steel has a metallic gray color with a slightly polished appearance. The defects are characterized by irregular, embedded scale particles that are pressed into the surface, creating a rough and uneven texture. The scale appears as dark, flaky patches within the smooth steel, contrasting sharply with the surrounding areas. The lighting highlights the texture and the embedded nature of the defects, casting subtle shadows and emphasizing the contrast between the smooth steel and the rough, scaly patches. The background is softly blurred to maintain focus on the steel surface, capturing the industrial and realistic nature of the defects in detail.|
	|scratches|A close-up view of a rolled steel surface showing scratches defects. The steel has a metallic gray finish with a slightly reflective sheen. The scratches are irregular, varying in length and depth, and are scattered across the surface. Some scratches are shallow and thin, while others are deeper and more pronounced, creating a rough texture. The lighting is angled to highlight the scratches, casting small shadows and emphasizing the texture and depth of the defects. The background is softly blurred to keep the focus on the steel surface, emphasizing the contrast between the smooth areas and the scratched defects. The image should convey a realistic industrial setting, with detailed and realistic depictions of the scratches.|

	|defect type|prompt|
	| :---: |:---------|
	|inclusion|"Close-up image of a rolled steel surface with visible inclusion defects. The steel has a smooth, polished metallic finish with dark, irregularly shaped inclusions embedded within the surface. These inclusions vary in size and shape, creating noticeable imperfections against the otherwise shiny steel. The lighting highlights the reflective nature of the steel, while also emphasizing the contrast between the smooth, clean areas and the embedded, uneven inclusions, making the defects clearly visible."|
	|oil_spot|"Close-up image of a rolled steel surface with visible oil spot defects. The steel has a smooth, polished metallic finish with dark, irregular oil spots scattered across the surface. The oil spots are prominent, with distinct edges and varying shapes and sizes, contrasting sharply against the shiny steel background. The lighting highlights the texture and sheen of the steel, while also emphasizing the presence and distribution of the oil spots."|
	|punching_hole|"Close-up image of a rolled steel surface with visible punching hole defects. The steel has a smooth, shiny metallic finish with several small, irregular holes scattered across the surface. The defects are clearly noticeable, with sharp edges and varying sizes, highlighting the imperfections on the otherwise uniform steel surface. The lighting emphasizes the texture and imperfections, providing a detailed view of the punching hole defects."|
	|water_spot|"Close-up image of a rolled steel surface with visible water spot defects. The steel has a smooth, reflective metallic finish with circular, light-colored water spots scattered across the surface. The water spots vary in size and shape, some with slightly blurred edges, giving a subtle contrast to the shiny steel background. The lighting captures the reflective nature of the steel and the slight discoloration and texture variation caused by the water spots, making the defects stand out."|
	|welding_line|"Close-up image of a rolled steel surface with visible welding line defects. The steel has a smooth, polished metallic finish with uneven, raised welding lines running across the surface. The welding lines are irregular, with jagged edges and varying thickness, disrupting the uniformity of the steel. The lighting highlights the texture differences, emphasizing the rough, imperfect weld lines against the smooth, shiny steel background, showcasing the imperfections clearly."|
	


- thought guide chain

	input1:  Suppose you are an AI expert with extensive knowledge in the field of industrial defect detection and large models. Please answer some questions for me.

	input2:  do you know rolled steel surface defects such as common types and their characterization?

	input3:  I need to generate an image of a crack defect using stable diffusion. Can you help me write a suitable prompt word?

	|defect type|prompt|
	|:---:|:-----------|
	|crazing|Create a highly detailed image of a crack defect on a rolled steel surface. The crack should be clearly visible, with irregular and jagged edges. It should run along the surface, displaying the typical characteristics of a longitudinal crack with possible branching and varying depths. The steel surface around the crack should show signs of wear and tear, with a slightly rough texture. The image should highlight the contrast between the smooth, undamaged areas of the steel and the distressed, cracked region.|
	|inclusion|Create a detailed image of an inclusion defect on a rolled steel surface. The inclusion should appear as a dark, irregularly shaped spot or cluster embedded within the steel. The surrounding steel surface should be smooth, but the inclusion area should have a contrasting texture, indicating impurities like oxides or sulfides. Highlight the differences between the clean steel and the defect area, showing how the inclusion disrupts the uniformity of the material. The image should emphasize the embedded nature of the inclusion and its impact on the steel's overall appearance and quality.|
	|patches|Create a detailed image of a patches defect on a rolled steel surface. The patches should appear as localized areas where the surface is either smoother or rougher compared to the surrounding material. These patches may vary in size and shape, appearing as irregular spots or streaks. The surrounding steel should have a consistent texture, while the patches should stand out, indicating areas of different surface quality. Highlight the contrast between the uniform steel surface and the defective patches, showing how these imperfections disrupt the overall appearance and quality of the material.|
	|pitted_surface|Create a detailed image of a pitted surface defect on a rolled steel surface. The defect should consist of numerous small, shallow depressions or pits scattered across the surface. The pits should vary in size and depth, creating an uneven texture. The surrounding steel surface should be relatively smooth, highlighting the contrast between the pitted areas and the undamaged metal. Emphasize the irregularity and distribution of the pits, showcasing the impact of the defect on the overall appearance and quality of the steel.|
	|rolled-in_scale|Generate a close-up image of a steel surface with a rolled-in scale defect. The surface should show irregular patches or lines of dark, brittle material embedded within the steel. These patches should contrast clearly with the smoother, shinier surrounding metal. The rolled-in scale should appear as rough, uneven areas with a distinct texture, indicating the presence of embedded oxide scale. The lighting should enhance the difference in texture and color between the scale and the steel, highlighting the defect|
	|scratches|Generate a close-up image of a steel surface with multiple scratch defects. The surface should exhibit numerous linear marks or grooves of varying lengths and depths, running in different directions. The scratches should have a rough texture and appear lighter or darker than the surrounding steel, depending on the angle of the light. Some scratches should be shallow and faint, while others should be deeper and more prominent. The lighting should accentuate the grooves and roughness, highlighting the contrast between the smooth steel and the scratched areas.|

	|defect type|prompt|
	| :---: |:---------|
	|inclusion|"Generate an image depicting an inclusion defect in rolled steel. The inclusion should appear as a dark, irregularly shaped particle embedded within the steel matrix. The size and shape of the inclusion should vary to simulate real-world variability."|
	|oil_spot|"Create an image showing an oil spot defect on the surface of rolled steel. The oil spot should have a distinct, irregular shape and a darker appearance compared to the surrounding steel surface. The size and location of the oil spot should vary to simulate different instances of this defect."|
	|punching_hole|"Generate an image illustrating a punching hole defect in rolled steel. The punching hole should appear as a circular or oval-shaped void in the steel surface, with clear edges and a uniform appearance. Vary the size and location of the punching hole to create different instances of this defect."|
	|water_spot|"Create an image depicting a water spot defect on rolled steel. The water spot should appear as a circular or irregularly shaped stain on the steel surface, with a lighter appearance compared to the surrounding steel. Vary the size and shape of the water spot to simulate different occurrences of this defect."|
	|welding_line|"Generate an image showing a welding line defect in rolled steel. The welding line should appear as a visible seam or line running along the surface of the steel, with variations in color and texture compared to the surrounding material. Vary the length and visibility of the welding line to create different instances of this defect."|

- image to text, with some labels

	input1:  Suppose you are an AI expert with extensive knowledge in the field of industrial defect detection and large models. Please answer some questions for me.

	input2:  do you know rolled steel surface defects such as common types and their characterization?

	input3:  This is a picture of a surface defect of rolled steel with a defect type of scratches. Can you describe this picture?(attach image)

	input4:  I want to use stable diffusion to generate scratches type images of rolled steel surface defects. Based on your description, can you help me write the prompt words? I hope the generated image is similar to the overall style of the image I just provided.

	|defect type|prompt|
	|:---:|:-----------|
	|crazing|Create a detailed image of a rolled steel surface exhibiting crack-type surface defects. The surface should display numerous small, irregular cracks, appearing as fine, dark lines that branch and spread across the surface. These cracks should form a dense network of fractures with varying lengths and orientations, giving the steel surface a fragmented appearance. Ensure the overall texture of the steel is visible, and the cracks create a pattern that reflects thermal stress, mechanical stress, or inherent material defects. The image should convey the characteristic appearance of cracks, similar to the reference provided.|
	|inclusion|Create a detailed image of a rolled steel surface with inclusion-type defects. The surface should show elongated, irregular dark lines or streaks embedded within the steel matrix, varying in length and thickness. These inclusions should disrupt the uniformity of the steel surface, indicating the presence of non-metallic particles or impurities. The overall texture of the steel should be visible, and the pattern of inclusions should reflect embedded foreign particles and surface disruption. The image should convey the characteristic appearance of inclusions similar to the reference provided.|
	|patches|Create a detailed image of a rolled steel surface exhibiting patch-type surface defects. The surface should display irregular, patchy areas with variations in discoloration or texture. These patches should appear as unevenly distributed dark and light regions, disrupting the uniform appearance of the steel. The patches should vary in size and shape, indicating inconsistencies in the surface treatment or material composition. The overall texture of the steel should be visible, with the patches creating a pattern that reflects surface contamination, improper heat treatment, or coating issues. The image should convey the characteristic appearance of patches similar to the reference provided.|
	|pitted_surface|Generate a detailed image of a rolled steel surface exhibiting pit-type surface defects. The surface should display numerous small, localized depressions scattered randomly, with varying sizes and depths. The defects should appear as dark spots or irregularities against the steel's metallic background. Ensure the overall texture of the steel is visible, and the pits create a rough and uneven appearance. The image should convey the characteristics of pitting, including surface erosion and material loss, similar to the provided reference.|
	|rolled-in_scale|Create a detailed image of a rolled steel surface exhibiting rolled-in scale surface defects. The surface should display irregular, embedded dark patches characteristic of scale that has been pressed into the steel during the rolling process. These defects should appear as uneven areas with varying sizes and shapes, disrupting the uniform appearance of the steel surface. Ensure the overall texture of the steel is visible, and the scale patches create a pattern that reflects inadequate descaling, improper rolling techniques, or environmental contamination. The image should convey the characteristic appearance of rolled-in scale, similar to the reference provided.|
	|scratches|Create a detailed image of a rolled steel surface exhibiting scratch-type surface defects. The surface should display several long, fine, linear marks that vary in length and depth, characteristic of scratches. These scratches should run parallel to each other and disrupt the smooth appearance of the steel surface. Ensure the overall texture of the steel is visible, and the scratches create a pattern that reflects mechanical abrasion, improper handling, defective equipment, or the presence of abrasive contaminants. The image should convey the characteristic appearance of scratches, similar to the reference provided.|

    |defect type|prompt|
	| :---: |:---------|
	|inclusion|"Create a detailed image of a rolled steel surface with inclusion-type defects. The surface should show elongated, irregular dark lines or streaks embedded within the steel matrix, varying in length and thickness. These inclusions should disrupt the uniformity of the steel surface, indicating the presence of non-metallic particles or impurities. The overall texture of the steel should be visible, and the pattern of inclusions should reflect embedded foreign particles and surface disruption. The image should convey the characteristic appearance of inclusions similar to the reference provided."|
	|oil_spot|"Generate an image of a rolled steel surface with visible oil spot defects. The steel surface should have a slightly textured, metallic appearance. The defects should appear as irregular, dark spots of varying sizes scattered across the steel surface. Ensure the spots are darker than the surrounding steel and have relatively well-defined but irregular edges, blending slightly with the steel surface. The lighting should highlight the contrast between the oil spots and the steel background to emphasize the presence of the defects."|
	|punching_hole|"Generate an image of a rolled steel surface with a visible punching hole defect. The steel surface should have a slightly textured, metallic appearance. The defect should appear as a dark, oval-shaped depression or hole with well-defined edges, contrasting against the lighter background of the steel. Ensure the lighting highlights the depth and boundaries of the hole to emphasize its presence on the steel surface."|
	|water_spot|"Generate an image of a rolled steel surface with visible water spot defects. The steel surface should have a slightly textured, metallic appearance. The defects should appear as irregular, dark streaks or smudges on the steel surface. Ensure the water spots are darker than the surrounding steel and have irregular edges that blend somewhat into the surrounding steel, creating a gradient effect. The lighting should highlight the contrast between the water spots and the steel background to emphasize the presence of the defects."|
	|welding_line|"Generate an image of a rolled steel surface with a welding line defect. The surface should appear mostly uniform and smooth, with a distinct, slightly darker horizontal line running across the middle of the image. The line should look irregular, indicating a welding defect, and contrast subtly with the surrounding steel. The texture of the line should appear either slightly raised or indented. Ensure the overall style and appearance of the image are similar to a high-resolution photograph, emphasizing the metallic quality and texture of the steel."|

- 结果

  - 可视化结果

  	![error](expansion_result.jpg)

  - 测试集准确率：
	|Prompt Method|5-shot|expansion|5-shot+expansion|fine-tuning|training|
	|:--------:|:---:|:---:|:---:|:---:|:---:|
	|label to text|0.595|0.778|0.782|0.82|0.933|
	|thought guide chain|0.595|0.843|0.827|0.853|0.933|
	|image to text|0.595|0.826|0.843|0.88|0.933|

  - 指标平均值：
	|Prompt Method|FID|IE|SSIM|
	|:--------:|:---:|:---:|:---:|
	|label to text|192.1|0.226|0.220|
	|thought guide chain|184.4|0.202|0.214|
	|image to text|179.2|0.212|0.210|

  - FID:
	|Prompt Method|class-1|class-2|class-3|class-4|class-5|class-6|
	|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
	|label to text|120.0|213.9|193.1|142.5|291.0|192.3|
	|thought guide chain|108.8|218.4|211.9|136.9|224.2|206.3|
	|image to text|123.2|236.2|188.3|137.4|208.3|181.8|
  - IE:
	|Prompt Method|class-1|class-2|class-3|class-4|class-5|class-6|
	|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
	|label to text|0.289|0.297|0.082|0.249|0.439|0.004|
	|thought guide chain|0.345|0.313|0.059|0.229|0.252|0.015|
	|image to text|0.326|0.380|0.111|0.196|0.253|0.005|
	|original image|0.009|0.190|0.005|0.005|0.331|0.003|

  - SSIM:
	|Prompt Method|class-1|class-2|class-3|class-4|class-5|class-6|
	|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
	|label to text|0.071|0.468|0.051|0.186|0.218|0.327|
	|thought guide chain|0.069|0.459|0.049|0.178|0.204|0.323|
	|image to text|0.071|0.444|0.052|0.171|0.197|0.324|

- 参数
  - stable diffusion基础参数
	```
	model:
	base_learning_rate: 1.0e-04
	target: ldm.models.diffusion.ddpm.LatentDiffusion
	params:
		linear_start: 0.00085
		linear_end: 0.0120
		num_timesteps_cond: 1
		log_every_t: 200
		timesteps: 1000
		first_stage_key: "jpg"
		cond_stage_key: "txt"
		image_size: 64
		channels: 4
		cond_stage_trainable: false   # Note: different from the one we trained before
		conditioning_key: crossattn
		monitor: val/loss_simple_ema
		scale_factor: 0.18215
		use_ema: False

		scheduler_config: # 10000 warmup steps
		target: ldm.lr_scheduler.LambdaLinearScheduler
		params:
			warm_up_steps: [ 10000 ]
			cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
			f_start: [ 1.e-6 ]
			f_max: [ 1. ]
			f_min: [ 1. ]

		unet_config:
		target: ldm.modules.diffusionmodules.openaimodel.UNetModel
		params:
			image_size: 32 # unused
			in_channels: 4
			out_channels: 4
			model_channels: 320
			attention_resolutions: [ 4, 2, 1 ]
			num_res_blocks: 2
			channel_mult: [ 1, 2, 4, 4 ]
			num_heads: 8
			use_spatial_transformer: True
			transformer_depth: 1
			context_dim: 768
			use_checkpoint: True
			legacy: False

		first_stage_config:
		target: ldm.models.autoencoder.AutoencoderKL
		params:
			embed_dim: 4
			monitor: val/rec_loss
			ddconfig:
			double_z: true
			z_channels: 4
			resolution: 256
			in_channels: 3
			out_ch: 3
			ch: 128
			ch_mult:
			- 1
			- 2
			- 4
			- 4
			num_res_blocks: 2
			attn_resolutions: []
			dropout: 0.0
			lossconfig:
			target: torch.nn.Identity

		cond_stage_config:
		target: ldm.modules.encoders.modules.FrozenCLIPEmbedder

	```
  - 调整的参数
	- scale
	  prompt的权重，实验选择20
	- strength
	  对init image的重建程度，1表示完全重建，实验选择0.3

3.2 Dalle-based Defect image generation

x',y' 5→5 vectors→5*n images

3.3 Automated Data Augmentation Framework

f(x',y,)→f'

**4.Case study**

4.1 dataset

4.2 visual analytics



4.3 prediction results （**对比的方法需要明确**）

a baseline:直接训练

b data augmentation:

c few-shot: pretraining [xxx]

d Our methods

4.4 Ablation experiment

**5.Conlusions**

