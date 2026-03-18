import sys
import struct
from vulkan import *

# Load SPIR-V shader binary
def load_shader(filename):
    with open(filename, "rb") as f:
        code = f.read()
    return code

def findMemoryType(physicalDevice, memoryTypeBits, properties):
        memoryProperties = vkGetPhysicalDeviceMemoryProperties(physicalDevice)

        # How does this search work?
        # See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
        for i, mt in enumerate(memoryProperties.memoryTypes):
            if memoryTypeBits & (1 << i) and (mt.propertyFlags & properties) == properties:
                return i

        return -1

def createBuffer(physicalDevice, device, bufferSize, buffer, bufferMemory):
    # We will now create a buffer. We will render the mandelbrot set into this buffer
    # in a computer shade later.
    bufferCreateInfo = VkBufferCreateInfo(
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=bufferSize,  # buffer size in bytes.
        usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # buffer is used as a storage buffer.
        sharingMode=VK_SHARING_MODE_EXCLUSIVE  # buffer is exclusive to a single queue family at a time.
    )

    buffer = vkCreateBuffer(device, bufferCreateInfo, None)

    # But the buffer doesn't allocate memory for itself, so we must do that manually.

    # First, we find the memory requirements for the buffer.
    memoryRequirements = vkGetBufferMemoryRequirements(device, buffer)

    # There are several types of memory that can be allocated, and we must choose a memory type that:
    # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
    # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
    #    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
    # Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
    # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
    # this flag.
    index = findMemoryType(physicalDevice, memoryRequirements.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    # Now use obtained memory requirements info to allocate the memory for the buffer.
    allocateInfo = VkMemoryAllocateInfo(
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=memoryRequirements.size,  # specify required memory.
        memoryTypeIndex=index
    )

    # allocate memory on device.
    bufferMemory = vkAllocateMemory(device, allocateInfo, None)

    # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
    vkBindBufferMemory(device, buffer, bufferMemory, 0)

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

# get the results into a numpy array here

print("Minimal Vulkan compute pipeline created successfully.")

# Cleanup
vkDestroyPipeline(device, pipeline, None)
vkDestroyPipelineLayout(device, pipeline_layout, None)
vkDestroyShaderModule(device, shader_module, None)
vkDestroyDevice(device, None)
vkDestroyInstance(instance, None)