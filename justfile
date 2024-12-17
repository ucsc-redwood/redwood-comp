
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/28.0.12674087/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=28 -c --vulkan-backend=y -v
    # xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/28.0.12433566/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=24 -c --vulkan-backend=y -v

set-android-v7a:
    xmake f -p android -a armeabi-v7a --ndk=~/Android/Sdk/ndk/28.0.12674087/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=24 -c --cuda-backend=n --vulkan-backend=y -v

set-jetson:
    xmake f -p linux -a arm64 -c --cuda-backend=y --vulkan-backend=y -v

set-default:
    xmake f -p linux -a x86_64 -c

run-utility:
    xmake r print-core-info
    xmake r check-affinity

run-bm-pc:
    xmake r bm-cifar-dense-cpu -d pc
    xmake r bm-cifar-sparse-cpu -d pc
    xmake r bm-tree-cpu -d pc

run-bm-android:
    xmake r bm-cifar-dense-vk
    xmake r bm-cifar-sparse-vk
    xmake r bm-tree-vk

run-bm-jetson:
    xmake r bm-cifar-dense-cuda -d jetson
    xmake r bm-cifar-sparse-cuda -d jetson
    xmake r bm-tree-cuda -d jetson


