import numpy as np
import VulkanCompute

if __name__ == '__main__':
    compute = VulkanCompute.VulkanCompute()

    # Define parameters for the compute shader
    # need to match the parameters in the shader code
    num_cameras = 6
    height = 1080
    width = 1920
    channels = 4
    workgroup_size = 16

    shader_file = 'lerp.spv'

    # pass the images from numpy to the shader here
    color_array = np.zeros((num_cameras, height, width, channels), dtype=np.float32)
    condition_array = np.zeros((num_cameras, height, width, channels), dtype=np.float32)
    pixel_array = np.zeros((num_cameras, height, width, channels), dtype=np.int32)

    accumulation_normalization = np.zeros((height, width, channels), dtype=np.float32)
    accumulation_normalization[:,:,0] = 1.0

    panorama = np.zeros((height, width, channels), dtype=np.float32)

    for n in range(num_cameras):
        # for c in range(channels):
        for x in range(width):
            r = x / (width-1.0)        
            for y in range(height):
                g = y / (height-1.0)        
                
                color_array[n,y,x,0] = r
                color_array[n,y,x,1] = g
                color_array[n,y,x,2] = n / (num_cameras-1.0)

        color_array[n,:,:,3] = 1.0

        VulkanCompute.VulkanCompute.SaveImage(color_array[n,:,:,:], f'color_{n}.png')

    #we might need to convert them to C,H,W

    compute.Setup(color_array, pixel_array, condition_array, accumulation_normalization, panorama, shader_file, workgroup_size, enable_validation_layers=True)

    print('Vulkan compute pipeline created successfully.')

    compute.Run()

    print('Vulkan compute pipeline run successfully.')
