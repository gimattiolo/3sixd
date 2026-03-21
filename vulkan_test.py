import sys
import os

import array
import math

import ctypes

import numpy as np

from PIL import Image

from vulkan import *

class VulkanCompute :

    def __init__(self) :

        self.num_cameras = 0
        self.height = 0
        self.width = 0
        self.channels = 0
        self.workgroup_size = 0

        self.app_info = None
        self.instance = None
        self.buffer_info = None
        self.shader_module = None
        self.descriptor_pool = None
        self.descriptor_set_layout = None
        self.pipeline_layout = None
        self.pipeline = None
        self.command_pool = None
        self.device = None
        self.debug_report_callback = None
        self.enable_validation_layers = False
        self.physical_device = None
        self.descriptor_set_layout_bindings = None
        self.shader_file = None

    # Load SPIR-V shader binary
    def load_shader(filename):
        with open(filename, "rb") as f:
            code = f.read()
        return code

    def FindMemoryType(self, memory_type_bits, properties):
            memory_properties = vkGetPhysicalDeviceMemoryProperties(self.physical_device)

            # How does this search work?
            # See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
            for i, mt in enumerate(memory_properties.memoryTypes):
                if memory_type_bits & (1 << i) and (mt.propertyFlags & properties) == properties:
                    return i

            return -1

    def CreateBuffer(self, buffer_size, pAllocator, pBuffer):
        # We will now create a buffer. We will render the mandelbrot set into this buffer
        # in a computer shade later.
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=buffer_size,  # buffer size in bytes.
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # buffer is used as a storage buffer.
            sharingMode=VK_SHARING_MODE_EXCLUSIVE  # buffer is exclusive to a single queue family at a time.
        )

        buffer = vkCreateBuffer(self.device, buffer_info, pAllocator, pBuffer)

        # But the buffer doesn't allocate memory for itself, so we must do that manually.

        # First, we find the memory requirements for the buffer.
        memory_requirements = vkGetBufferMemoryRequirements(self.device, buffer)

        # There are several types of memory that can be allocated, and we must choose a memory type that:
        # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
        # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
        #    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
        # Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
        # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        # this flag.
        index = self.FindMemoryType(memory_requirements.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        # Now use obtained memory requirements info to allocate the memory for the buffer.
        allocate_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memory_requirements.size,  # specify required memory.
            memoryTypeIndex=index
        )

        # allocate memory on device.
        buffer_memory = vkAllocateMemory(self.device, allocate_info, None)

        # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
        vkBindBufferMemory(self.device, buffer, buffer_memory, 0)

        return buffer, buffer_memory

    def CreateImage2D(self, physical_device, device, width, height):
        # We will now create a buffer. We will render the mandelbrot set into this buffer
        # in a computer shade later.
        buffer_info = VkImageCreateInfo(
            sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=VK_IMAGE_TYPE_2D,
            format=VK_FORMAT_R32G32B32A32_SFLOAT,
            extent=VkExtent3D(
                width=width,
                height=height,
                depth=1
            ),
            mipLevels=1,
            arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )

        image = vkCreateImage(device, buffer_info, None)

        # But the image doesn't allocate memory for itself, so we must do that manually.

        # First, we find the memory requirements for the image.
        memory_requirements = vkGetImageMemoryRequirements(device, image)

        # There are several types of memory that can be allocated, and we must choose a memory type that:
        # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
        # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
        #    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
        # Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
        # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        # this flag.
        # index = FindMemoryType(physical_device, memory_requirements.memoryTypeBits,
        #                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)

        #index = FindMemoryType(physical_device, memory_requirements.memoryTypeBits,
        #                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)


        #assert index != -1, "Failed to find suitable memory type for image"
        
        # Now use obtained memory requirements info to allocate the memory for the image.
        allocate_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memory_requirements.size,  # specify required memory.
            #memoryTypeIndex=index
        )

        # allocate memory on device.
        image_memory = vkAllocateMemory(device, allocate_info, None)

        # Now associate that allocated memory with the image. With that, the image is backed by actual memory.
        vkBindImageMemory(device, image, image_memory, memoryOffset=0)

        # subresource_range = VkImageSubresourceRange(
        #     aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
        #     baseMipLevel=0,
        #     levelCount=1,
        #     baseArrayLayer=0,
        #     layerCount=1)

        # components = VkComponentMapping(
        #     r=VK_COMPONENT_SWIZZLE_IDENTITY,
        #     g=VK_COMPONENT_SWIZZLE_IDENTITY,
        #     b=VK_COMPONENT_SWIZZLE_IDENTITY,
        #     a=VK_COMPONENT_SWIZZLE_IDENTITY)

        # imageview_create = VkImageViewCreateInfo(
        #     sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        #     image=image,
        #     flags=0,
        #     viewType=VK_IMAGE_VIEW_TYPE_2D,
        #     format=surface_format.format,
        #     components=components,
        #     subresourceRange=subresource_range)

        # image_views.append(vkCreateImageView(logical_device, imageview_create, None))            

        return image, image_memory

    def CreateDescriptorSetLayout(self):

        descriptor_set_layout_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(self.descriptor_set_layout_bindings),  
            pBindings=self.descriptor_set_layout_bindings
        )

        # Create the descriptor set layout.
        self.descriptor_set_layout = vkCreateDescriptorSetLayout(self.device, descriptor_set_layout_info, pAllocator=None)

    def CreateComputePipeline(self):
        # We create a compute pipeline here.

        # Create a shader module. A shader module basically just encapsulates some shader code.
        with open(self.shader_file, 'rb') as comp:
            code = comp.read()

            shader_module_info = VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code),
                pCode=code
            )

            self.shader_module = vkCreateShaderModule(self.device, shader_module_info, None)

        # Now let us actually create the compute pipeline.
        # A compute pipeline is very simple compared to a graphics pipeline.
        # It only consists of a single stage with a compute shader.
        # So first we specify the compute shader stage, and it's entry point(main).
        pipeline_shader_stage_info = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName='main'
        )

        # The pipeline layout allows the pipeline to access descriptor sets.
        # So we just specify the descriptor set layout we created earlier.
        
        # pipeline_layout_info = VkPipelineLayoutCreateInfo(
        #     sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
        # )
        
        pipeline_layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout]
        )
        self.pipeline_layout = vkCreatePipelineLayout(self.device, pipeline_layout_info, None)

        pipeline_info = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=pipeline_shader_stage_info,
            layout=self.pipeline_layout
        )

        # Now, we finally create the compute pipeline.
        pipelines = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, pipeline_info, None)
        if len(pipelines) == 1:
            self.pipeline = pipelines[0]
        else:
            raise Exception("Could not create compute pipeline")

    def CreateCommandBuffer(self, queue_family_index):
        # We are getting closer to the end. In order to send commands to the device(GPU),
        # we must first record commands into a command buffer.
        # To allocate a command buffer, we must first create a command pool. So let us do that.
        command_pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=0,
            # the queue family of this command pool. All command buffers allocated from this command pool,
            # must be submitted to queues of this family ONLY.
            queueFamilyIndex=queue_family_index
        )

        self.command_pool = vkCreateCommandPool(self.device, command_pool_info, None)

        # Now allocate a command buffer from the command pool.
        command_buffer_allocate_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            # if the command buffer is primary, it can be directly submitted to queues.
            # A secondary buffer has to be called from some primary command buffer, and cannot be directly
            # submitted to a queue. To keep things simple, we use a primary command buffer.
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        self.command_buffer = vkAllocateCommandBuffers(self.device, command_buffer_allocate_info)[0]

        # Now we shall start recording commands into the newly allocated command buffer.
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            # the buffer is only submitted and used once in this application.
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vkBeginCommandBuffer(self.command_buffer, begin_info)

        # We need to bind a pipeline, AND a descriptor set before we dispatch.
        # The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        vkCmdBindPipeline(self.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vkCmdBindDescriptorSets(self.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                                0, 1, [self.descriptor_set], 0, None)

        # Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        # The number of workgroups is specified in the arguments.
        # If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        vkCmdDispatch(self.command_buffer,
                        int(math.ceil(self.width / float(self.workgroup_size))),  # int for py2 compatible
                        int(math.ceil(self.height / float(self.workgroup_size))),  # int for py2 compatible
                        1)

        vkEndCommandBuffer(self.command_buffer)


    def CreateDescriptorSet(self,buffer, buffer_size):
        # So we will allocate a descriptor set here.
        # But we need to first create a descriptor pool to do that.

        # Our descriptor pool can only allocate a single storage buffer.
        descriptor_pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1
        )

        descriptor_pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,  # we only need to allocate one descriptor set from the pool.
            poolSizeCount=1,
            pPoolSizes=descriptor_pool_size
        )

        # create descriptor pool.
        self.descriptor_pool = vkCreateDescriptorPool(self.device, descriptor_pool_info, None)

        # With the pool allocated, we can now allocate the descriptor set.
        descriptor_set_allocate_info = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_set_layout]
        )

        # allocate descriptor set.
        self.descriptor_set = vkAllocateDescriptorSets(self.device, descriptor_set_allocate_info)[0]


    def UpdateWriteDescriptorSet(self, descriptor_buffer_info, binding):
        write_descriptor_set = VkWriteDescriptorSet(
            sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.descriptor_set,
            dstBinding=binding, # write to the specified binding.
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=descriptor_buffer_info
        )

        # perform the update of the descriptor set.
        vkUpdateDescriptorSets(self.device, descriptorWriteCount=1, pDescriptorWrites=[write_descriptor_set], descriptorCopyCount=0, pDescriptorCopies=None)

    # def UpdateReadDescriptorSet(self, descriptor_buffer_info, binding):
    #     read_descriptor_set = VkReadDescriptorSet(
    #         sType=VK_STRUCTURE_TYPE_READ_DESCRIPTOR_SET,
    #         dstSet=self.descriptor_set,
    #         dstBinding=binding, # write to the specified binding.
    #         descriptorCount=1,
    #         descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    #         pBufferInfo=descriptor_buffer_info
    #     )

    #     # perform the update of the descriptor set.
    #     vkUpdateDescriptorSets(self.device, descriptorWriteCount=1, pDescriptorWrites=[read_descriptor_set], descriptorCopyCount=0, pDescriptorCopies=None)

    def RunCommandBuffer(self):
        # Now we shall finally submit the recorded command buffer to a queue.
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,  # submit a single command buffer
            pCommandBuffers=[self.command_buffer]  # the command buffer to submit.
        )

        # We create a fence.
        fence_info = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=0
        )
        fence = vkCreateFence(self.device, fence_info, None)

        # We submit the command buffer on the queue, at the same time giving a fence.
        vkQueueSubmit(self.queue, 1, submit_info, fence)

        # The command will not have finished executing until the fence is signalled.
        # So we wait here.
        # We will directly after this read our buffer from the GPU,
        # and we will not be sure that the command has finished executing unless we wait for the fence.
        # Hence, we use a fence here.
        vkWaitForFences(self.device, 1, [fence], VK_TRUE, 100000000000)

        vkDestroyFence(self.device, fence, None)

    def GetBufferAsNumpy(self, buffer_memory, buffer_size, H, W, C):
        # Map the buffer memory, so that we can read from it on the CPU.
        p_mapped_memory = vkMapMemory(self.device, buffer_memory, 0, buffer_size, 0)

        # Get the color data from the buffer, and cast it to bytes.
        # We save the data to a vector.

        pa = np.frombuffer(p_mapped_memory, np.float32)

        # Done reading, so unmap.
        vkUnmapMemory(self.device, buffer_memory)

        pa = pa.reshape((H, W, C))

        return pa

    def Cleanup(self):
        # Clean up all Vulkan Resources.

        if self.enable_validation_layers:
            # destroy callback.
            func = vkGetInstanceProcAddr(self.instance, 'vkDestroyDebugReportCallbackEXT')
            if func == ffi.NULL:
                raise Exception("Could not load vkDestroyDebugReportCallbackEXT")
            if self.debug_report_callback:
                func(self.instance, self.debug_report_callback, None)

        for binding, buf, buf_memory, buf_size in self.buffer_info:
            if buf_memory:
                vkFreeMemory(self.device, buf_memory, None)
            if buf:
                vkDestroyBuffer(self.device, buf, None)
        if self.shader_module:
            vkDestroyShaderModule(self.device, self.shader_module, None)
        if self.descriptor_pool:
            vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
        if self.descriptor_set_layout:
            vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
        if self.pipeline_layout:
            vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        if self.pipeline:
            vkDestroyPipeline(self.device, self.pipeline, None)
        if self.command_pool:
            vkDestroyCommandPool(self.device, self.command_pool, None)
        if self.device:
            vkDestroyDevice(self.device, None)
        if self.instance:
            vkDestroyInstance(self.instance, None)

    def InitializeBuffer(self, numpy_array, buffer_memory, buffer_size):

        # Ensure array is contiguous in memory
        if not numpy_array.flags['C_CONTIGUOUS']:
            numpy_array = np.ascontiguousarray(numpy_array)

        ffi_buffer = vkMapMemory(self.device, buffer_memory, offset=0, size=buffer_size, flags=0)
        src_bytes = numpy_array.tobytes()
        ffi_buffer[:len(src_bytes)] = src_bytes
        vkUnmapMemory(self.device, buffer_memory)

    def SaveImage(a, file_path):
        a = np.copy(a)
        # assume image array is in range [0,1]
        a *= 255.0
        a = a.astype(np.uint8)
        image = Image.fromarray(a)
        if os.path.exists(file_path):
            os.remove(file_path)
        image.save(file_path)

    def Run(self) :

        # Finally, run the recorded command buffer.

        self.RunCommandBuffer()

        binding_id = 4

        binding, panorama_buffer, panorama_buffer_memory, buffer_size = self.buffer_info[binding_id]

        assert binding == binding_id, f"Expected binding {binding_id} but got {binding}"
        
        # get the results into a numpy array here
        output_image = self.GetBufferAsNumpy(panorama_buffer_memory, buffer_size, self.height, self.width, self.channels)

        # Now we save the acquired color data to a .png.
        VulkanCompute.SaveImage(output_image, 'test.png')

    def Setup(self, shader_file, num_cameras, height, width, channels, workgroup_size):

        self.shader_file = shader_file

        self.num_cameras = num_cameras
        self.height = height
        self.width = width
        self.channels = channels
        self.workgroup_size = workgroup_size

        # Create Vulkan instance
        self.app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="MinimalCompute".encode(),
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="NoEngine".encode(),
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION_1_0
        )

        instance_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=self.app_info
        )

        try:
            self.instance = vkCreateInstance(instance_info, None)
        except VkErrorInitializationFailed:
            print("Failed to create Vulkan instance")
            sys.exit(1)

        # Enumerate physical devices (GPUs)
        devices = vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            print("No Vulkan-compatible GPU found.")
        else:
            print(f"Found {len(devices)} Vulkan device(s):")
            for device in devices:
                props = vkGetPhysicalDeviceProperties(device)
                # print(f" - {props.deviceName.decode('utf-8')}")
                print(f"{props.deviceName}")

        # Pick first physical device
        physical_devices = vkEnumeratePhysicalDevices(self.instance)
        self.physical_device = physical_devices[0]

        # Find compute queue family
        queue_family_index = None
        for i, props in enumerate(vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)):
            if props.queueFlags & VK_QUEUE_COMPUTE_BIT:
                queue_family_index = i
                break

        if queue_family_index is None:
            print("No compute queue found")
            sys.exit(1)

        # Create logical device and queue
        queue_priority = 1.0
        device_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[
                VkDeviceQueueCreateInfo(
                    sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    queueFamilyIndex=queue_family_index,
                    queueCount=1,
                    pQueuePriorities=[queue_priority]
                )
            ]
        )

        self.device = vkCreateDevice(self.physical_device, device_info, None)
        self.queue = vkGetDeviceQueue(self.device, queue_family_index, 0)

        # pass the images from numpy to the shader here
        color_array = np.zeros((self.num_cameras, self.height, self.width, self.channels), dtype=np.float32)
        condition_array = np.zeros((self.num_cameras, self.height, self.width, self.channels), dtype=np.float32)
        pixel_array = np.zeros((self.num_cameras, self.height, self.width, self.channels), dtype=np.float32)

        accumulation_normalization = np.zeros((self.height, self.width, self.channels), dtype=np.float32)
        accumulation_normalization[:,:,0] = 1.0

        panorama = np.zeros((self.height, self.width, self.channels), dtype=np.float32)

        for n in range(self.num_cameras):
            # for c in range(self.channels):
            for x in range(self.width):
                r = x / (self.width-1.0)        
                for y in range(self.height):
                    g = y / (self.height-1.0)        
                    
                    color_array[n,y,x,0] = r
                    color_array[n,y,x,1] = g
                    color_array[n,y,x,2] = n / (self.num_cameras-1.0)

            color_array[n,:,:,3] = 1.0

            VulkanCompute.SaveImage(color_array[n,:,:,:], f'color_{n}.png')

        #we might need to convert them to C,H,W

        pixel = array.array('f', [0, 0, 0, 0]) # vec4
        address, num_bytes = pixel.buffer_info()

        buffer_size = self.width * self.height * self.channels * num_bytes  
        buffer_array_size = self.num_cameras * self.width * self.height * self.channels * num_bytes

        accumulation_normalization_buffer, accumulation_normalization_buffer_memory = self.CreateBuffer(buffer_size, pAllocator=None, pBuffer=None)
        color_array_buffer, color_array_buffer_memory = self.CreateBuffer(buffer_array_size, pAllocator=None, pBuffer=None)
        condition_array_buffer, condition_array_buffer_memory = self.CreateBuffer(buffer_array_size, pAllocator=None, pBuffer=None)
        pixel_array_buffer, pixel_array_buffer_memory = self.CreateBuffer(buffer_array_size, pAllocator=None, pBuffer=None)
        panorama_buffer, panorama_buffer_memory = self.CreateBuffer(buffer_size, pAllocator=None, pBuffer=None)

        self.InitializeBuffer(accumulation_normalization, accumulation_normalization_buffer_memory, buffer_size)
        self.InitializeBuffer(color_array, color_array_buffer_memory, buffer_array_size)
        self.InitializeBuffer(condition_array, condition_array_buffer_memory, buffer_array_size)
        self.InitializeBuffer(pixel_array, pixel_array_buffer_memory, buffer_array_size)
        self.InitializeBuffer(panorama, panorama_buffer_memory, buffer_size)

        # Here we specify a descriptor set layout. This allows us to bind our descriptors to
        # resources in the shader.

        # Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
        # 0. This binds to
        #   layout(std140, binding = 0) buffer buf
        # in the compute shader.

        self.buffer_info = [
            (0, accumulation_normalization_buffer, accumulation_normalization_buffer_memory, buffer_size),
            (1, color_array_buffer, color_array_buffer_memory, buffer_array_size),
            (2, condition_array_buffer, condition_array_buffer_memory, buffer_array_size),
            (3, pixel_array_buffer, pixel_array_buffer_memory, buffer_array_size),
            (4, panorama_buffer, panorama_buffer_memory, buffer_size),
        ]

        self.descriptor_set_layout_bindings = [None] * len(self.buffer_info)

        for binding, buf, buf_memory, buf_size in self.buffer_info:
            self.descriptor_set_layout_bindings[binding] = VkDescriptorSetLayoutBinding(
                binding, 
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 
                descriptorCount=1, 
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT
            )

        self.CreateDescriptorSetLayout()

        self.CreateComputePipeline()

        self.CreateDescriptorSet(panorama_buffer, buffer_size)

        # Next, we need to connect our actual storage buffer with the descriptor.
        # We use vkUpdateDescriptorSets() to update the descriptor set.

        for binding, buf, buf_memory, buf_size in self.buffer_info:
            # Specify the buffer to bind to the descriptor.
            descriptor_buffer_info = VkDescriptorBufferInfo(
                buffer=buf,
                offset=0,
                range=buf_size
            )
            self.UpdateWriteDescriptorSet(descriptor_buffer_info, binding)

        self.CreateCommandBuffer(queue_family_index)


if __name__ == "__main__":
    compute = VulkanCompute()

    # Define parameters for the compute shader
    # need to match the parameters in the shader code
    num_cameras = 6
    height = 1080
    width = 1920
    channels = 4
    workgroup_size = 16

    shader_file = "lerp.spv"
    compute.Setup(shader_file, num_cameras, height, width, channels, workgroup_size)

    print("Vulkan compute pipeline created successfully.")

    compute.Run()

    print("Vulkan compute pipeline run successfully.")

    compute.Cleanup()

