import sys
import struct
from vulkan import *

# Load SPIR-V shader binary
def load_shader(filename):
    with open(filename, "rb") as f:
        code = f.read()
    return code

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
shader_code = load_shader("shader.spv")
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

print("Minimal Vulkan compute pipeline created successfully.")

# Cleanup
vkDestroyPipeline(device, pipeline, None)
vkDestroyPipelineLayout(device, pipeline_layout, None)
vkDestroyShaderModule(device, shader_module, None)
vkDestroyDevice(device, None)
vkDestroyInstance(instance, None)