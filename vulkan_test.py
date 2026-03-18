import sys
from vulkan import *

def main():
    # Create Vulkan application info
    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="Python Vulkan Example",
        applicationVersion=VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=VK_MAKE_VERSION(1, 0, 0),
        apiVersion=VK_API_VERSION_1_0
    )

    # Create Vulkan instance
    create_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info
    )

    try:
        instance = vkCreateInstance(create_info, None)
    except VkErrorInitializationFailed:
        print("Failed to initialize Vulkan. Ensure Vulkan SDK is installed.")
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

    # Clean up
    vkDestroyInstance(instance, None)

if __name__ == "__main__":
    main()