
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/28.0.12433566/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=24 -c

set-android-v7a:
    xmake f -p android -a armeabi-v7a --ndk=~/Android/Sdk/ndk/28.0.12433566/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=24 -c

set-jetson:
    xmake f -p linux -a arm64 -c

set-default:
    xmake f -p linux -a x86_64 -c

pkg:
    xmake package -o ./local-package-repo vk-comp
    xmake package -o ./local-package-repo thread-pool

run-utility:
    xmake r print-core-info
    xmake r check-affinity
