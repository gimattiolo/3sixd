import sys
import struct

import numpy as np
import array
import math

from vulkan import *

# Load SPIR-V shader binary
def load_shader(filename):
    with open(filename, "rb") as f:
        code = f.read()
    return code

def FindMemoryType(physical_device, memory_type_bits, properties):
        memory_properties = vkGetPhysicalDeviceMemoryProperties(physical_device)

        # How does this search work?
        # See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
        for i, mt in enumerate(memory_properties.memoryTypes):
            if memory_type_bits & (1 << i) and (mt.propertyFlags & properties) == properties:
                return i

        return -1

def CreateBuffer(physical_device, device, buffer_size):
    # We will now create a buffer. We will render the mandelbrot set into this buffer
    # in a computer shade later.
    buffer_create_info = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=buffer_size,  # buffer size in bytes.
        usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # buffer is used as a storage buffer.
        sharingMode=VK_SHARING_MODE_EXCLUSIVE  # buffer is exclusive to a single queue family at a time.
    )

    buffer = vkCreateBuffer(device, buffer_create_info, None)

    # But the buffer doesn't allocate memory for itself, so we must do that manually.

    # First, we find the memory requirements for the buffer.
    memory_requirements = vkGetBufferMemoryRequirements(device, buffer)

    # There are several types of memory that can be allocated, and we must choose a memory type that:
    # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
    # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
    #    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
    # Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
    # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
    # this flag.
    index = FindMemoryType(physical_device, memory_requirements.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    # Now use obtained memory requirements info to allocate the memory for the buffer.
    allocate_info = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=memory_requirements.size,  # specify required memory.
        memoryTypeIndex=index
    )

    # allocate memory on device.
    buffer_memory = vkAllocateMemory(device, allocate_info, None)

    # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
    vkBindBufferMemory(device, buffer, buffer_memory, 0)

    return buffer, buffer_memory


def CreateDescriptorSetLayout(device):
    # Here we specify a descriptor set layout. This allows us to bind our descriptors to
    # resources in the shader.

    # Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
    # 0. This binds to
    #   layout(std140, binding = 0) buffer buf
    # in the compute shader.

    descriptor_set_layoutBinding = VkDescriptorSetLayoutBinding(
        binding=0,
        descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=1,
        stageFlags=VK_SHADER_STAGE_COMPUTE_BIT
    )

    descriptor_set_layout_create_info = VkDescriptorSetLayoutCreateInfo(
        sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        bindingCount=1,  # only a single binding in this descriptor set layout.
        pBindings=descriptor_set_layoutBinding
    )

    # Create the descriptor set layout.
    descriptor_set_layout = vkCreateDescriptorSetLayout(device, descriptor_set_layout_create_info, None)
    return descriptor_set_layout

def CreateComputePipeline(device, descriptor_set_layout):
    # We create a compute pipeline here.

    # Create a shader module. A shader module basically just encapsulates some shader code.
    with open('mandelbrot_compute.spv', 'rb') as comp:
        code = comp.read()

        create_info = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code
        )

        compute_shader_module = vkCreateShaderModule(device, create_info, None)

    # Now let us actually create the compute pipeline.
    # A compute pipeline is very simple compared to a graphics pipeline.
    # It only consists of a single stage with a compute shader.
    # So first we specify the compute shader stage, and it's entry point(main).
    shader_stage_create_info = VkPipelineShaderStageCreateInfo(
        sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=VK_SHADER_STAGE_COMPUTE_BIT,
        module=compute_shader_module,
        pName='main'
    )

    # The pipeline layout allows the pipeline to access descriptor sets.
    # So we just specify the descriptor set layout we created earlier.
    pipeline_layout_create_info = VkPipelineLayoutCreateInfo(
        sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        setLayoutCount=1,
        pSetLayouts=[descriptor_set_layout]
    )
    pipeline_layout = vkCreatePipelineLayout(device, pipeline_layout_create_info, None)

    pipeline_create_info = VkComputePipelineCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=shader_stage_create_info,
        layout=pipeline_layout
    )

    # Now, we finally create the compute pipeline.
    pipelines = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, pipeline_create_info, None)
    if len(pipelines) == 1:
        return pipelines[0]
    else:
        raise Exception("Could not create compute pipeline")

def CreateCommandBuffer(device, queue_family_index, width, height, workgroup_size):
    # We are getting closer to the end. In order to send commands to the device(GPU),
    # we must first record commands into a command buffer.
    # To allocate a command buffer, we must first create a command pool. So let us do that.
    command_pool_create_info = VkCommandPoolCreateInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        flags=0,
        # the queue family of this command pool. All command buffers allocated from this command pool,
        # must be submitted to queues of this family ONLY.
        queueFamilyIndex=queue_family_index
    )

    command_pool = vkCreateCommandPool(device, command_pool_create_info, None)

    # Now allocate a command buffer from the command pool.
    command_buffer_allocate_info = VkCommandBufferAllocateInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool=command_pool,
        # if the command buffer is primary, it can be directly submitted to queues.
        # A secondary buffer has to be called from some primary command buffer, and cannot be directly
        # submitted to a queue. To keep things simple, we use a primary command buffer.
        level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1
    )

    command_buffer = vkAllocateCommandBuffers(device, command_buffer_allocate_info)[0]

    # Now we shall start recording commands into the newly allocated command buffer.
    begin_info = VkCommandBufferBeginInfo(
        sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        # the buffer is only submitted and used once in this application.
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )
    vkBeginCommandBuffer(command_buffer, begin_info)

    # We need to bind a pipeline, AND a descriptor set before we dispatch.
    # The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                            0, 1, [descriptor_set], 0, None)

    # Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
    # The number of workgroups is specified in the arguments.
    # If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
    vkCmdDispatch(command_buffer,
                    int(math.ceil(width / float(workgroup_size))),  # int for py2 compatible
                    int(math.ceil(height / float(workgroup_size))),  # int for py2 compatible
                    1)

    vkEndCommandBuffer(command_buffer)

    return command_buffer, command_pool

def CreateDescriptorSet(device, descriptor_set_layout, buffer, buffer_size):
    # So we will allocate a descriptor set here.
    # But we need to first create a descriptor pool to do that.

    # Our descriptor pool can only allocate a single storage buffer.
    descriptor_pool_size = VkDescriptorPoolSize(
        type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        descriptorCount=1
    )

    descriptor_pool_create_info = VkDescriptorPoolCreateInfo(
        sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        maxSets=1,  # we only need to allocate one descriptor set from the pool.
        poolSizeCount=1,
        pPoolSizes=descriptor_pool_size
    )

    # create descriptor pool.
    descriptorPool = vkCreateDescriptorPool(device, descriptor_pool_create_info, None)

    # With the pool allocated, we can now allocate the descriptor set.
    descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
        sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool=descriptor_pool,
        descriptorSetCount=1,
        pSetLayouts=[descriptor_set_layout]
    )

    # allocate descriptor set.
    descriptor_set = vkAllocateDescriptorSets(device, descriptorSetAllocateInfo)[0]

    # Next, we need to connect our actual storage buffer with the descrptor.
    # We use vkUpdateDescriptorSets() to update the descriptor set.

    # Specify the buffer to bind to the descriptor.
    descriptor_buffer_Info = VkDescriptorBufferInfo(
        buffer=buffer,
        offset=0,
        range=buffer_size
    )

    write_descriptor_set = VkWriteDescriptorSet(
        sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        dstSet=descriptor_set,
        dstBinding=0,  # write to the first, and only binding.
        descriptorCount=1,
        descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        pBufferInfo=descriptor_buffer_Info
    )

    # perform the update of the descriptor set.
    vkUpdateDescriptorSets(device, 1, [write_descriptor_set], 0, None)

    return descriptor_set, descriptor_pool

def RunCommandBuffer(device, command_buffer, queue):
    # Now we shall finally submit the recorded command buffer to a queue.
    submit_info = VkSubmitInfo(
        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        commandBufferCount=1,  # submit a single command buffer
        pCommandBuffers=[command_buffer]  # the command buffer to submit.
    )

    # We create a fence.
    fence_create_info = VkFenceCreateInfo(
        sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        flags=0
    )
    fence = vkCreateFence(device, fence_create_info, None)

    # We submit the command buffer on the queue, at the same time giving a fence.
    vkQueueSubmit(queue, 1, submit_info, fence)

    # The command will not have finished executing until the fence is signalled.
    # So we wait here.
    # We will directly after this read our buffer from the GPU,
    # and we will not be sure that the command has finished executing unless we wait for the fence.
    # Hence, we use a fence here.
    vkWaitForFences(device, 1, [fence], VK_TRUE, 100000000000)

    vkDestroyFence(device, fence, None)

def GetOutputImage(device, buffer_memory, buffer_size, H, W):
    # Map the buffer memory, so that we can read from it on the CPU.
    p_mapped_memory = vkMapMemory(device, buffer_memory, 0, buffer_size, 0)

    # Get the color data from the buffer, and cast it to bytes.
    # We save the data to a vector.

    pa = np.frombuffer(p_mapped_memory, np.float32)
    pa = pa.reshape((H, W, 4))
    pa *= 255

    # Done reading, so unmap.
    vkUnmapMemory(device, buffer_memory)

    return pa

def Cleanup(instance, 
    buffer_memory, 
    buffer, 
    compute_shader_module, 
    descriptor_pool, 
    descriptor_set_layout, 
    pipeline_layout, 
    pipeline, 
    command_pool, 
    device, 
    debug_report_callback,
    enable_validation_layers):
    # Clean up all Vulkan Resources.

    if enable_validation_layers:
        # destroy callback.
        func = vkGetInstanceProcAddr(instance, 'vkDestroyDebugReportCallbackEXT')
        if func == ffi.NULL:
            raise Exception("Could not load vkDestroyDebugReportCallbackEXT")
        if debug_report_callback:
            func(instance, debug_report_callback, None)

    if buffer_memory:
        vkFreeMemory(device, buffer_memory, None)
    if buffer:
        vkDestroyBuffer(device, buffer, None)
    if compute_shader_module:
        vkDestroyShaderModule(device, compute_shader_module, None)
    if descriptor_pool:
        vkDestroyDescriptorPool(device, descriptor_pool, None)
    if descriptor_set_layout:
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, None)
    if pipeline_layout:
        vkDestroyPipelineLayout(device, pipeline_layout, None)
    if pipeline:
        vkDestroyPipeline(device, pipeline, None)
    if command_pool:
        vkDestroyCommandPool(device, command_pool, None)
    if device:
        vkDestroyDevice(device, None)
    if instance:
        vkDestroyInstance(instance, None)

# Create Vulkan instance
app_info = VkApplicationInfo(
    sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName="MinimalCompute".encode(),
    applicationVersion=VK_MAKE_VERSION(1, 0, 0),
    pEngineName="NoEngine".encode(),
    engineVersion=VK_MAKE_VERSION(1, 0, 0),
    apiVersion=VK_API_VERSION_1_0
)

instance_info = VkInstanceCreateInfo(
    sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pApplicationInfo=app_info
)

try:
    instance = vkCreateInstance(instance_info, None)
except VkErrorInitializationFailed:
    print("Failed to create Vulkan instance")
    sys.exit(1)

 # Enumerate physical devices (GPUs)
devices = vkEnumeratePhysicalDevices(instance)
if not devices:
    print("No Vulkan-compatible GPU found.")
else:
    print(f"Found {len(devices)} Vulkan device(s):")
    for device in devices:
        props = vkGetPhysicalDeviceProperties(device)
        # print(f" - {props.deviceName.decode('utf-8')}")
        print(f"{props.deviceName}")

# Pick first physical device
physical_devices = vkEnumeratePhysicalDevices(instance)
physical_device = physical_devices[0]

# Find compute queue family
queue_family_index = None
for i, props in enumerate(vkGetPhysicalDeviceQueueFamilyProperties(physical_device)):
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

device = vkCreateDevice(physical_device, device_info, None)
queue = vkGetDeviceQueue(device, queue_family_index, 0)

# Create shader module
shader_code = load_shader("lerp.spv")
shader_module_info = VkShaderModuleCreateInfo(
    sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    codeSize=len(shader_code),
    pCode=shader_code
)
shader_module = vkCreateShaderModule(device, shader_module_info, None)

# Create compute pipeline
pipeline_layout_info = VkPipelineLayoutCreateInfo(
    sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
)
pipeline_layout = vkCreatePipelineLayout(device, pipeline_layout_info, None)

stage_info = VkPipelineShaderStageCreateInfo(
    sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    stage=VK_SHADER_STAGE_COMPUTE_BIT,
    module=shader_module,
    pName="main".encode()
)

pipeline_info = VkComputePipelineCreateInfo(
    sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    stage=stage_info,
    layout=pipeline_layout
)

pipeline = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, [pipeline_info], None)[0]

# pass the images from numpy to the shader here

HEIGHT = 1080
WIDTH = 1920
N = 6
CHANNELS = 4
WORKGROUP_SIZE = 16

panorama_image = np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.float32)

color_image_array = np.zeros((HEIGHT, WIDTH, CHANNELS, N), dtype=np.float32)
condition_image_array = np.zeros((HEIGHT, WIDTH, CHANNELS, N), dtype=np.float32)
pixel_image_array = np.zeros((HEIGHT, WIDTH, CHANNELS, N), dtype=np.float32)
accumulation_normalization_image = np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.float32)

#we might need to convert them to C,H,W

# layout(binding = 0, rgba32f) writeonly uniform image2D panorama_image;
# layout(binding = 1, rgba32f) readonly uniform image2DArray color_image_array;
# layout(binding = 2, rgba32f) readonly uniform image2DArray condition_image_array;
# layout(binding = 3, rgba32f) readonly uniform image2DArray pixel_image_array;
# layout(binding = 4, rgba32f) readonly uniform image2D accumulation_normalization_image;

layoutBindings = [ 
    VkDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
    VkDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
    VkDescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
    VkDescriptorSetLayoutBinding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
    VkDescriptorSetLayoutBinding(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
]

layoutInfo = VkDescriptorSetLayoutCreateInfo(
    sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    bindingCount=len(layoutBindings),  
)
computeDescriptorSetLayout = vkCreateDescriptorSetLayout(device, layoutInfo, None)

# The mandelbrot set will be rendered to this buffer.
# The memory that backs the buffer is bufferMemory.
buffer = None
buffer_memory = None
buffer_size = 0

# size of buffer in bytes.
pixel = array.array('f', [0, 0, 0, 0])
buffer_size = pixel.buffer_info()[1] * pixel.itemsize * WIDTH * HEIGHT

buffer, buffer_memory = CreateBuffer(physical_device, device, buffer_size)
compute_descriptor_set_layout = CreateDescriptorSetLayout(device)
descriptor_set, descriptor_pool = CreateDescriptorSet(device, compute_descriptor_set_layout, buffer, buffer_size)
pipeline = CreateComputePipeline(device, compute_descriptor_set_layout)
command_buffer, command_pool = CreateCommandBuffer(device, queue_family_index, WIDTH, HEIGHT, WORKGROUP_SIZE)

# Finally, run the recorded command buffer.
RunCommandBuffer(device, command_buffer, queue)

# get the results into a numpy array here
output_image = GetOutputImage(device, buffer_memory, buffer_size, HEIGHT, WIDTH)

print("Minimal Vulkan compute pipeline created successfully.")

# Cleanup
debug_report_callback = None
enable_validation_layers = False
Cleanup(instance, buffer_memory, buffer, shader_module, descriptor_pool, compute_descriptor_set_layout, pipeline_layout, pipeline, command_pool, device, debug_report_callback, enable_validation_layers)
