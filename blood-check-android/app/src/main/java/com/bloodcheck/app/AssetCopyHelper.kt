package com.bloodcheck.app

import android.content.Context
import java.io.File

class AssetCopyHelper(private val context: Context) {

    fun ensureAppDataReady(): File {
        val targetRoot = File(context.filesDir, "blood_check_data")
        copyAssetTree("model", File(targetRoot, "model"))
        copyAssetTree("origin_model", File(targetRoot, "origin_model"))
        copyAssetTree("baseinfo", File(targetRoot, "baseinfo"))
        copyAssetTree("spectral_data", File(targetRoot, "spectral_data"))
        return targetRoot
    }

    private fun copyAssetTree(assetPath: String, target: File) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            copyAssetFile(assetPath, target)
            return
        }

        if (!target.exists()) {
            target.mkdirs()
        }

        children.forEach { child ->
            copyAssetTree("$assetPath/$child", File(target, child))
        }
    }

    private fun copyAssetFile(assetPath: String, target: File) {
        if (target.exists() && target.length() > 0L) {
            return
        }

        target.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            target.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }
}
