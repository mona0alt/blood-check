package com.bloodcheck.app

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothProfile
import android.bluetooth.BluetoothStatusCodes
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.util.ArrayDeque
import java.util.UUID

enum class BleMonitorMode {
    STATUS_ONLY,
    COLLECTING
}

enum class BleConnectionState {
    IDLE,
    SCANNING,
    CONNECTING,
    CONNECTED,
    DISCONNECTED,
    ERROR
}

data class BleConnectionStatus(
    val state: BleConnectionState,
    val message: String
)

data class BleSignalSnapshot(
    val red: List<Double>,
    val infrared: List<Double>
) {
    val size: Int get() = minOf(red.size, infrared.size)
}

data class BleRawValues(
    val c1: Int,
    val c2: Int,
    val c3: Int,
    val red: Double,
    val infrared: Double
)

object BleRawSampleParser {
    private val arrayRegex = Regex("""array:\s*\[([^\]]+)]""")
    private val integerRegex = Regex("""-?\d+""")

    fun parse(text: String): BleRawValues? {
        val match = arrayRegex.find(text) ?: return null
        val numbers = match.groupValues[1]
            .split(",")
            .map { it.trim() }
            .takeIf { values -> values.size == 5 && values.all { integerRegex.matches(it) } }
            ?.map { it.toInt() }
            ?: return null

        return BleRawValues(
            c1 = numbers[0],
            c2 = numbers[1],
            c3 = numbers[2],
            red = numbers[3].toDouble(),
            infrared = numbers[4].toDouble()
        )
    }
}

class BleSignalBuffer(private val maxPoints: Int) {
    private val lock = Any()
    private val redSignals = ArrayDeque<Double>()
    private val infraredSignals = ArrayDeque<Double>()

    val size: Int
        get() = synchronized(lock) {
            minOf(redSignals.size, infraredSignals.size)
        }

    init {
        require(maxPoints > 0) { "maxPoints must be positive" }
    }

    fun add(values: BleRawValues): Int = synchronized(lock) {
        redSignals.addLast(values.red)
        infraredSignals.addLast(values.infrared)
        trimLocked()
        minOf(redSignals.size, infraredSignals.size)
    }

    fun snapshot(): BleSignalSnapshot = synchronized(lock) {
        val minSize = minOf(redSignals.size, infraredSignals.size)
        if (minSize <= 0) {
            BleSignalSnapshot(emptyList(), emptyList())
        } else {
            BleSignalSnapshot(
                redSignals.toList().takeLast(minSize),
                infraredSignals.toList().takeLast(minSize)
            )
        }
    }

    fun clear() {
        synchronized(lock) {
            redSignals.clear()
            infraredSignals.clear()
        }
    }

    private fun trimLocked() {
        while (redSignals.size > maxPoints) redSignals.removeFirst()
        while (infraredSignals.size > maxPoints) infraredSignals.removeFirst()
    }
}

class BleMonitorManager(
    private val context: Context,
    onStatus: (String) -> Unit,
    onSampleCountChanged: (Int, Long) -> Unit,
    onError: (String) -> Unit,
    mode: BleMonitorMode = BleMonitorMode.COLLECTING,
    warmupMillis: Long = 0L,
    private val scanTimeoutMillis: Long = 12_000L,
    private val onConnectionStatus: (BleConnectionStatus) -> Unit = {},
    private val onRawSamples: (List<BleRawValues>, Long) -> Unit = { _, _ -> }
) {
    companion object {
        private const val TAG = "BloodCheckBle"
        private const val TARGET_DEVICE_NAME = "Nordic_UART"
        private const val MAX_SIGNAL_POINTS = 6000
        private val NUS_SERVICE_UUID: UUID = UUID.fromString("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
        private val NUS_TX_CHAR_UUID: UUID = UUID.fromString("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
        private val CLIENT_CONFIG_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val signalBuffer = BleSignalBuffer(MAX_SIGNAL_POINTS)
    private val pendingText = StringBuilder()
    private val collectionStateLock = Any()

    private var adapter: BluetoothAdapter? = null
    private var gatt: BluetoothGatt? = null
    private var txCharacteristic: BluetoothGattCharacteristic? = null
    private var scanning = false
    private var stoppedByClient = false
    private var firstDataElapsedMs = 0L
    private var warmupLogged = false
    @Volatile private var monitorMode: BleMonitorMode = mode
    @Volatile private var collectionWarmupMillis: Long = warmupMillis
    @Volatile private var onStatusCallback: (String) -> Unit = onStatus
    @Volatile private var onSampleCountChangedCallback: (Int, Long) -> Unit = onSampleCountChanged
    @Volatile private var onErrorCallback: (String) -> Unit = onError

    val isReadyToCollect: Boolean
        get() = gatt != null && txCharacteristic != null

    init {
        require(collectionWarmupMillis >= 0L) { "warmupMillis must not be negative" }
        require(scanTimeoutMillis > 0L) { "scanTimeoutMillis must be positive" }
    }

    private val scanTimeoutRunnable = Runnable {
        if (!scanning) return@Runnable
        postConnectionStatus(BleConnectionState.ERROR, "未发现 Nordic_UART")
        stopScan()
        postError("未发现 Nordic_UART")
    }

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device ?: return
            val name = result.scanRecord?.deviceName ?: device.name
            if (name != TARGET_DEVICE_NAME) return
            Log.i(TAG, "scan found $name address=${device.address}")
            postStatus("发现设备，正在连接...")
            postConnectionStatus(BleConnectionState.CONNECTING, "正在连接 Nordic_UART")
            stopScan()
            try {
                gatt = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    device.connectGatt(context, false, gattCallback, BluetoothDeviceTransportLe)
                } else {
                    device.connectGatt(context, false, gattCallback)
                }
            } catch (e: SecurityException) {
                postConnectionStatus(BleConnectionState.ERROR, "缺少蓝牙连接权限")
                postError("缺少蓝牙连接权限")
            }
        }

        override fun onScanFailed(errorCode: Int) {
            scanning = false
            mainHandler.removeCallbacks(scanTimeoutRunnable)
            postConnectionStatus(BleConnectionState.ERROR, "蓝牙扫描失败: $errorCode")
            postError("蓝牙扫描失败: $errorCode")
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            Log.i(TAG, "connection state status=$status newState=$newState")
            if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                postStatus("设备已连接，正在发现服务...")
                postConnectionStatus(BleConnectionState.CONNECTING, "已连接，正在发现服务")
                val discovering = try {
                    gatt.discoverServices()
                } catch (e: SecurityException) {
                    false
                }
                if (!discovering) {
                    postConnectionStatus(BleConnectionState.ERROR, "蓝牙服务发现启动失败")
                    postError("蓝牙服务发现启动失败")
                }
                return
            }

            if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                closeGatt()
                if (stoppedByClient) return
                postConnectionStatus(BleConnectionState.DISCONNECTED, "蓝牙设备已断开")
                postError("蓝牙设备已断开")
                return
            }

            if (status != BluetoothGatt.GATT_SUCCESS) {
                closeGatt()
                if (stoppedByClient) return
                postConnectionStatus(BleConnectionState.ERROR, "蓝牙连接异常: $status")
                postError("蓝牙连接异常: $status")
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            Log.i(TAG, "services discovered status=$status")
            if (status != BluetoothGatt.GATT_SUCCESS) {
                postConnectionStatus(BleConnectionState.ERROR, "蓝牙服务发现失败: $status")
                postError("蓝牙服务发现失败: $status")
                return
            }
            val tx = gatt.getService(NUS_SERVICE_UUID)?.getCharacteristic(NUS_TX_CHAR_UUID)
            if (tx == null) {
                postConnectionStatus(BleConnectionState.ERROR, "未找到 Nordic UART TX 特征")
                postError("未找到 Nordic UART TX 特征")
                return
            }
            txCharacteristic = tx

            if (monitorMode == BleMonitorMode.STATUS_ONLY) {
                postConnectionStatus(BleConnectionState.CONNECTED, "蓝牙连接正常")
                return
            }

            enableDataNotifications(gatt, tx)
        }

        @Deprecated("Deprecated by Android 13 callback overload")
        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
            @Suppress("DEPRECATION")
            handleNotification(characteristic.value)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            handleNotification(value)
        }

        override fun onDescriptorWrite(
            gatt: BluetoothGatt,
            descriptor: BluetoothGattDescriptor,
            status: Int
        ) {
            if (descriptor.uuid != CLIENT_CONFIG_UUID) return
            Log.i(TAG, "notification descriptor write status=$status")
            if (status == BluetoothGatt.GATT_SUCCESS) {
                postConnectionStatus(BleConnectionState.CONNECTED, "蓝牙连接正常，正在采集")
                postStatus("已开始接收实时数据")
            } else {
                postConnectionStatus(BleConnectionState.ERROR, "蓝牙通知描述符写入失败: $status")
                postError("蓝牙通知描述符写入失败: $status")
            }
        }
    }

    fun beginCollecting(
        warmupMillis: Long,
        onStatus: (String) -> Unit,
        onSampleCountChanged: (Int, Long) -> Unit,
        onError: (String) -> Unit
    ): Boolean {
        require(warmupMillis >= 0L) { "warmupMillis must not be negative" }
        onStatusCallback = onStatus
        onSampleCountChangedCallback = onSampleCountChanged
        onErrorCallback = onError
        monitorMode = BleMonitorMode.COLLECTING
        collectionWarmupMillis = warmupMillis
        resetCollectionBuffers()

        val activeGatt = gatt
        val tx = txCharacteristic
        if (activeGatt != null && tx != null) {
            return enableDataNotifications(activeGatt, tx)
        }
        return scanning || activeGatt != null
    }

    fun start() {
        stoppedByClient = false
        firstDataElapsedMs = 0L
        warmupLogged = false
        if (monitorMode == BleMonitorMode.COLLECTING) {
            clearSignals()
        } else {
            mainHandler.post { onSampleCountChangedCallback(0, 0L) }
        }
        val manager = context.getSystemService(Context.BLUETOOTH_SERVICE) as android.bluetooth.BluetoothManager
        adapter = manager.adapter
        val bluetoothAdapter = adapter
        if (bluetoothAdapter == null || !bluetoothAdapter.isEnabled) {
            postConnectionStatus(BleConnectionState.ERROR, "请先开启手机蓝牙")
            postError("请先开启手机蓝牙")
            return
        }
        val scanner = bluetoothAdapter.bluetoothLeScanner
        if (scanner == null) {
            postConnectionStatus(BleConnectionState.ERROR, "当前设备不支持 BLE 扫描")
            postError("当前设备不支持 BLE 扫描")
            return
        }
        postStatus("正在扫描 Nordic_UART...")
        postConnectionStatus(BleConnectionState.SCANNING, "正在扫描 Nordic_UART")
        scanning = true
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()
        try {
            scanner.startScan(null, settings, scanCallback)
            mainHandler.postDelayed(scanTimeoutRunnable, scanTimeoutMillis)
        } catch (e: SecurityException) {
            scanning = false
            postConnectionStatus(BleConnectionState.ERROR, "缺少蓝牙扫描权限")
            postError("缺少蓝牙扫描权限")
        }
    }

    fun stop(silent: Boolean = false) {
        stoppedByClient = true
        stopScan()
        closeGatt()
        if (!silent) {
            postStatus("实时监测已停止")
        }
    }

    fun snapshot(): BleSignalSnapshot {
        return signalBuffer.snapshot()
    }

    private fun enableDataNotifications(gatt: BluetoothGatt, tx: BluetoothGattCharacteristic): Boolean {
        try {
            val notificationEnabled = gatt.setCharacteristicNotification(tx, true)
            if (!notificationEnabled) {
                postConnectionStatus(BleConnectionState.ERROR, "蓝牙通知启用失败")
                postError("蓝牙通知启用失败")
                return false
            }

            val descriptor = tx.getDescriptor(CLIENT_CONFIG_UUID)
            Log.i(TAG, "tx characteristic found descriptor=${descriptor != null}")
            if (descriptor == null) {
                postConnectionStatus(BleConnectionState.ERROR, "未找到蓝牙通知描述符")
                postError("未找到蓝牙通知描述符")
                return false
            }

            val descriptorWriteSucceeded = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                gatt.writeDescriptor(
                    descriptor,
                    BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                ) == BluetoothStatusCodes.SUCCESS
            } else {
                @Suppress("DEPRECATION")
                descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                @Suppress("DEPRECATION")
                gatt.writeDescriptor(descriptor)
            }
            if (!descriptorWriteSucceeded) {
                postConnectionStatus(BleConnectionState.ERROR, "蓝牙通知描述符写入失败")
                postError("蓝牙通知描述符写入失败")
                return false
            }
        } catch (e: SecurityException) {
            postConnectionStatus(BleConnectionState.ERROR, "缺少蓝牙通知权限")
            postError("缺少蓝牙通知权限")
            return false
        }
        return true
    }

    private fun handleNotification(value: ByteArray) {
        if (monitorMode != BleMonitorMode.COLLECTING) return
        val chunk = value.toString(Charsets.UTF_8)
        var droppedBufferPreview: String? = null
        val parsed = synchronized(pendingText) {
            pendingText.append(chunk)
            val rows = mutableListOf<BleRawValues>()
            while (true) {
                val text = pendingText.toString()
                val match = Regex("""array:\s*\[([^\]]+)]""").find(text) ?: break
                BleRawSampleParser.parse(match.value)?.let { rows.add(it) }
                pendingText.delete(0, match.range.last + 1)
                val sumIndex = pendingText.indexOf("}")
                if (sumIndex >= 0) {
                    pendingText.delete(0, sumIndex + 1)
                }
            }
            if (pendingText.length > 512) {
                droppedBufferPreview = pendingText.take(120).toString()
                pendingText.clear()
            }
            rows
        }
        droppedBufferPreview?.let { preview ->
            Log.w(TAG, "dropping unparsable buffer=$preview")
        }
        if (parsed.isEmpty()) {
            Log.d(TAG, "notification chunk not parsed len=${value.size} text=${chunk.take(80)}")
            return
        }

        val now = SystemClock.elapsedRealtime()
        var shouldLogWarmupDrop = false
        val stableElapsedMillis = synchronized(collectionStateLock) {
            if (firstDataElapsedMs == 0L) {
                firstDataElapsedMs = now
            }
            val elapsedMillis = now - firstDataElapsedMs - collectionWarmupMillis
            if (elapsedMillis < 0L && !warmupLogged) {
                warmupLogged = true
                shouldLogWarmupDrop = true
            }
            elapsedMillis
        }
        if (stableElapsedMillis < 0L) {
            if (shouldLogWarmupDrop) {
                Log.i(TAG, "warmup active, dropping initial samples for ${collectionWarmupMillis}ms")
            }
            return
        }

        var count = signalBuffer.size
        parsed.forEach { values ->
            count = signalBuffer.add(values)
        }
        onRawSamples(parsed, now)
        if (count <= 5 || count % 50 == 0) {
            Log.i(TAG, "parsedSamples=$count stableElapsedMs=$stableElapsedMillis last=${parsed.last()}")
        }
        mainHandler.post { onSampleCountChangedCallback(count, stableElapsedMillis) }
    }

    private fun clearSignals() {
        resetCollectionBuffers()
        mainHandler.post { onSampleCountChangedCallback(0, 0L) }
    }

    private fun resetCollectionBuffers() {
        signalBuffer.clear()
        synchronized(pendingText) {
            pendingText.clear()
        }
        synchronized(collectionStateLock) {
            firstDataElapsedMs = 0L
            warmupLogged = false
        }
    }

    private fun stopScan() {
        mainHandler.removeCallbacks(scanTimeoutRunnable)
        if (!scanning) return
        try {
            adapter?.bluetoothLeScanner?.stopScan(scanCallback)
        } catch (e: SecurityException) {
            Log.w(TAG, "missing scan permission while stopping scan", e)
        }
        scanning = false
    }

    private fun closeGatt() {
        gatt?.close()
        gatt = null
        txCharacteristic = null
    }

    private fun postStatus(message: String) {
        mainHandler.post { onStatusCallback(message) }
    }

    private fun postError(message: String) {
        mainHandler.post { onErrorCallback(message) }
    }

    private fun postConnectionStatus(state: BleConnectionState, message: String) {
        mainHandler.post { onConnectionStatus(BleConnectionStatus(state, message)) }
    }
}

private val BluetoothDeviceTransportLe: Int
    get() = 2
