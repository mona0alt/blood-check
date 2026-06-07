package com.bloodcheck.app

import android.Manifest
import android.os.Build

object BlePermissionPolicy {
    fun requiredPermissions(sdkInt: Int = Build.VERSION.SDK_INT): Array<String> {
        return if (sdkInt >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT
            )
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
    }

    fun missingPermissions(
        sdkInt: Int = Build.VERSION.SDK_INT,
        isPermissionGranted: (String) -> Boolean
    ): Array<String> {
        return requiredPermissions(sdkInt)
            .filterNot(isPermissionGranted)
            .toTypedArray()
    }

    fun deniedStatusMessage(sdkInt: Int = Build.VERSION.SDK_INT): String {
        return if (sdkInt >= Build.VERSION_CODES.S) {
            "附近设备权限未授权"
        } else {
            "定位权限未授权"
        }
    }

    fun requestMessage(sdkInt: Int = Build.VERSION.SDK_INT): String {
        return if (sdkInt >= Build.VERSION_CODES.S) {
            "需要附近设备权限才能连接蓝牙检测设备"
        } else {
            "需要定位权限才能扫描蓝牙检测设备"
        }
    }

    fun settingsMessage(sdkInt: Int = Build.VERSION.SDK_INT): String {
        return if (sdkInt >= Build.VERSION_CODES.S) {
            "请在系统设置中打开本应用的附近设备权限，然后返回继续检测"
        } else {
            "请在系统设置中打开本应用的定位权限，然后返回继续检测"
        }
    }

    fun requiresLocationService(sdkInt: Int = Build.VERSION.SDK_INT): Boolean {
        return sdkInt < Build.VERSION_CODES.S
    }
}
