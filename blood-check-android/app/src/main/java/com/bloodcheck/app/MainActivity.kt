package com.bloodcheck.app

import android.content.Intent
import android.content.res.ColorStateList
import android.content.pm.PackageManager
import android.os.Build
import android.graphics.PorterDuff
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.ImageView
import android.widget.RadioButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.ViewModelProvider
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "BloodCheckMain"
        // 仅用于调试包便捷测试；release 会自动走手动输入1
        private const val FIXED_PATIENT_ID = ""
        private const val REQUEST_BLE_PERMISSIONS = 1001
        private const val MIN_LIVE_SIGNAL_ROWS = 200
        private const val LIVE_WARMUP_MILLIS = 3_000L
        private const val LIVE_STABLE_WINDOW_MILLIS = 10_000L
        private const val BLE_STATUS_RETRY_MILLIS = 15_000L
    }

    private lateinit var viewModel: PredictViewModel

    private lateinit var rootMain: View
    private lateinit var statusBarSpacer: View
    private lateinit var tvHeaderTitle: TextView
    private lateinit var layoutHeaderActions: View
    private lateinit var btnDataCollect: View
    private lateinit var btnSettings: View
    private lateinit var scrollHome: View
    private lateinit var scrollMine: View

    private lateinit var tvPatientIdRow: TextView
    private lateinit var tvPatientName: TextView
    private lateinit var ivPatientGenderIcon: ImageView
    private lateinit var tvPatientGender: TextView
    private lateinit var tvPatientAge: TextView
    private lateinit var viewBleStatusDot: View
    private lateinit var tvBleStatusText: TextView

    private lateinit var chartMonitor: HemoglobinMonitorChartView
    private lateinit var viewLegendHb: View
    private lateinit var viewLegendLac: View

    private lateinit var tvDataHb: TextView
    private lateinit var tvDataLac: TextView
    private lateinit var tvDataPi: TextView
    private lateinit var tvDataSpo2: TextView
    private lateinit var tvDataPao2: TextView
    private lateinit var tvDataHr: TextView
    private lateinit var tvDataFo2Hb: TextView
    private lateinit var tvDataGlucose: TextView
    private lateinit var tvDataK: TextView
    private lateinit var tvDataNa: TextView
    private lateinit var btnResetAlarm: MaterialButton

    private lateinit var tvMonitoringPatientId: TextView
    private lateinit var layoutBottomInput: View
    private lateinit var btnStartMonitor: MaterialButton
    private lateinit var layoutMonitorActions: View
    private lateinit var btnStopMonitor: MaterialButton
    private lateinit var btnExport: MaterialButton

    private lateinit var bottomNav: View
    private lateinit var navHome: View
    private lateinit var navMine: View
    private lateinit var tvNavHome: TextView
    private lateinit var tvNavMine: TextView
    private lateinit var ivNavHome: ImageView
    private lateinit var ivNavMine: ImageView

    private lateinit var layoutMineContent: View
    private lateinit var tvMineEmpty: TextView
    private lateinit var tvMineValPatientId: TextView
    private lateinit var tvMineValUnified: TextView
    private lateinit var tvMineValHospital: TextView
    private lateinit var tvMineValName: TextView
    private lateinit var tvMineValGender: TextView
    private lateinit var tvMineValAge: TextView
    private lateinit var tvMineValTemp: TextView
    private lateinit var tvMineValHr: TextView
    private lateinit var tvMineValSpo2: TextView
    private lateinit var tvMineValPao2: TextView
    private lateinit var tvMineValO2hb: TextView

    private val handler = Handler(Looper.getMainLooper())
    private val monitorPollRunnable = Runnable {
        if (!isMonitoring) return@Runnable
        if (liveInferenceInFlight) return@Runnable
        val id = currentPatientId()
        if (id.isEmpty()) return@Runnable
        val snapshot = bleMonitorManager?.snapshot()
        if (snapshot == null || snapshot.size < MIN_LIVE_SIGNAL_ROWS) {
            pendingInferenceSpectrumRange = null
            if (liveMonitorGate.isOpen) {
                Toast.makeText(this, "实时数据不足，等待采集...", Toast.LENGTH_SHORT).show()
            }
            scheduleNextMonitorPoll()
            return@Runnable
        }
        val redSnapshot = snapshot.red
        val infraredSnapshot = snapshot.infrared
        val monitorToken = activeMonitorSessionToken
        val requestId = nextLiveInferenceRequestId()
        liveInferenceInFlight = true
        pendingLiveInferenceRequestId = requestId
        pendingInferencePatientId = id
        pendingInferenceMonitorToken = monitorToken
        pendingInferenceSpectrumRange = null
        submitMonitorFileIo(
            errorMessage = "监测文件状态读取失败",
            disableWritesOnFailure = false,
            skipWhenFileWritesDisabled = false
        ) {
            val spectrumRange = activeMonitorFileSession
                ?.takeIf { it.patientId == id && it.monitorToken == monitorToken }
                ?.batcher
                ?.currentRange()
            handler.post {
                if (
                    !isActiveMonitorSession(id, monitorToken) ||
                    pendingLiveInferenceRequestId != requestId
                ) {
                    if (pendingLiveInferenceRequestId == requestId) {
                        clearLiveInferenceState()
                    }
                    return@post
                }
                pendingInferenceSpectrumRange = spectrumRange
                viewModel.predictLiveForMonitoring(id, redSnapshot, infraredSnapshot, requestId)
            }
        }
    }

    private var isMonitoring: Boolean = false
    private var hbAlarm: Boolean = false
    private var currentTabHome: Boolean = true
    private var monitorModel: PatientMonitorModel? = null
    private var monitoringSessionPatientId: String? = null
    private var bleMonitorManager: BleMonitorManager? = null
    private var bleStatusManager: BleMonitorManager? = null
    private var pendingBleReadyAction: (() -> Unit)? = null
    private var blePermissionRequestInFlight: Boolean = false
    private var blePermissionDialogShowing: Boolean = false
    private var bleLocationDialogShowing: Boolean = false
    private val liveMonitorGate = LiveMonitorGate(MIN_LIVE_SIGNAL_ROWS, LIVE_STABLE_WINDOW_MILLIS)
    private val bleStatusRetryRunnable = Runnable { startBleStatusMonitor() }
    private val monitorFileExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val patientDataFileStore: PatientDataFileStore by lazy {
        PatientDataFileStore.fromContext(this)
    }
    private val patientRecordCsvWriter = PatientRecordCsvWriter()
    private val patientZipExporter = PatientDataZipExporter()
    @Volatile private var activeMonitorFileSession: MonitorFileSession? = null
    @Volatile private var activeMonitorSessionToken: Long = 0L
    @Volatile private var monitorFileWritesDisabled: Boolean = false
    @Volatile private var lastMonitorFileIoToastAtMillis: Long = 0L
    private var liveInferenceInFlight: Boolean = false
    private var liveInferenceRequestSequence: Long = 0L
    private var pendingLiveInferenceRequestId: Long? = null
    private var pendingInferencePatientId: String? = null
    private var pendingInferenceSpectrumRange: BleSpectrumRange? = null
    private var pendingInferenceMonitorToken: Long? = null

    private val hbSeries = mutableListOf<ChartPoint>()
    private val lacSeries = mutableListOf<ChartPoint>()
    private var lastHbChartTime: Long = 0L
    private var lastLacChartTime: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowCompat.getInsetsController(window, window.decorView)?.isAppearanceLightStatusBars = true

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        viewModel = ViewModelProvider(this)[PredictViewModel::class.java]

        bindViews()
        applyWindowInsets()

        setupPatientIdSource()

        navHome.setOnClickListener { selectTab(home = true) }
        navMine.setOnClickListener { selectTab(home = false) }
        btnDataCollect.setOnClickListener { showDataCollectionDialog() }
        btnSettings.setOnClickListener { showSettingsDialog() }
        btnStartMonitor.setOnClickListener { startMonitoringFromUi() }
        btnStopMonitor.setOnClickListener { stopMonitoring() }
        btnExport.setOnClickListener { exportCurrentPatientData() }
        btnResetAlarm.setOnClickListener { resetHbAlarm() }

        viewModel.monitorState.observe(this) { state ->
            when (state) {
                is MonitorUiState.Success -> {
                    val requestId = state.requestId
                    if (!isCurrentLiveInferenceRequest(requestId)) {
                        Log.w(TAG, "ignore stale monitor success requestId=${state.requestId}")
                        return@observe
                    }
                    onMonitorPredictionSuccess(state.response, requestId ?: return@observe)
                    clearLiveInferenceState()
                    if (isMonitoring) {
                        scheduleNextMonitorPoll()
                    }
                }
                is MonitorUiState.Error -> {
                    val requestId = state.requestId
                    if (!isCurrentLiveInferenceRequest(requestId)) {
                        Log.w(TAG, "ignore stale monitor error requestId=${state.requestId}")
                        return@observe
                    }
                    clearLiveInferenceState()
                    if (isMonitoring) {
                        Toast.makeText(this, state.message, Toast.LENGTH_SHORT).show()
                        scheduleNextMonitorPoll()
                    }
                }
                else -> Unit
            }
        }

        selectTab(home = true)
        updateMonitoringUi()
        updateHasPatientIdUi()

        rootMain.post {
            restoreMonitorStateFromStorage()
            startBleStatusMonitor()
        }
    }

    override fun onResume() {
        super.onResume()
        if (!::tvBleStatusText.isInitialized) return
        val pendingAction = pendingBleReadyAction
        if (pendingAction != null && hasBlePermissions() && isBleLocationServiceReady()) {
            pendingBleReadyAction = null
            handler.post { pendingAction.invoke() }
            return
        }
        if (isMonitoring || bleStatusManager != null) return
        if (hasBlePermissions() && isBleLocationServiceReady()) {
            handler.post { startBleStatusMonitor() }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        nextMonitorSessionToken()
        clearLiveInferenceState()
        handler.removeCallbacks(monitorPollRunnable)
        handler.removeCallbacks(bleStatusRetryRunnable)
        bleMonitorManager?.stop(silent = true)
        bleStatusManager?.stop(silent = true)
        flushActiveMonitorSpectrumAsync(clearSession = false)
        monitorFileExecutor.shutdown()
    }

    private fun bindViews() {
        rootMain = findViewById(R.id.rootMain)
        statusBarSpacer = findViewById(R.id.statusBarSpacer)
        tvHeaderTitle = findViewById(R.id.tvHeaderTitle)
        layoutHeaderActions = findViewById(R.id.layoutHeaderActions)
        btnDataCollect = findViewById(R.id.btnDataCollect)
        btnSettings = findViewById(R.id.btnSettings)
        scrollHome = findViewById(R.id.scrollHome)
        scrollMine = findViewById(R.id.scrollMine)

        tvPatientIdRow = findViewById(R.id.tvPatientIdRow)
        tvPatientName = findViewById(R.id.tvPatientName)
        ivPatientGenderIcon = findViewById(R.id.ivPatientGenderIcon)
        tvPatientGender = findViewById(R.id.tvPatientGender)
        tvPatientAge = findViewById(R.id.tvPatientAge)
        viewBleStatusDot = findViewById(R.id.viewBleStatusDot)
        tvBleStatusText = findViewById(R.id.tvBleStatusText)

        chartMonitor = findViewById(R.id.chartMonitor)
        viewLegendHb = findViewById(R.id.viewLegendHb)
        viewLegendLac = findViewById(R.id.viewLegendLac)

        tvDataHb = findViewById(R.id.tvDataHb)
        tvDataLac = findViewById(R.id.tvDataLac)
        tvDataPi = findViewById(R.id.tvDataPi)
        tvDataSpo2 = findViewById(R.id.tvDataSpo2)
        tvDataPao2 = findViewById(R.id.tvDataPao2)
        tvDataHr = findViewById(R.id.tvDataHr)
        tvDataFo2Hb = findViewById(R.id.tvDataFo2Hb)
        tvDataGlucose = findViewById(R.id.tvDataGlucose)
        tvDataK = findViewById(R.id.tvDataK)
        tvDataNa = findViewById(R.id.tvDataNa)
        btnResetAlarm = findViewById(R.id.btnResetAlarm)

        tvMonitoringPatientId = findViewById(R.id.tvMonitoringPatientId)
        layoutBottomInput = findViewById(R.id.layoutBottomInput)
        btnStartMonitor = findViewById(R.id.btnStartMonitor)
        layoutMonitorActions = findViewById(R.id.layoutMonitorActions)
        btnStopMonitor = findViewById(R.id.btnStopMonitor)
        btnExport = findViewById(R.id.btnExport)

        bottomNav = findViewById(R.id.bottomNav)
        navHome = findViewById(R.id.navHome)
        navMine = findViewById(R.id.navMine)
        tvNavHome = findViewById(R.id.tvNavHome)
        tvNavMine = findViewById(R.id.tvNavMine)
        ivNavHome = findViewById(R.id.ivNavHome)
        ivNavMine = findViewById(R.id.ivNavMine)

        layoutMineContent = findViewById(R.id.layoutMineContent)
        tvMineEmpty = findViewById(R.id.tvMineEmpty)
        tvMineValPatientId = findViewById(R.id.tvMineValPatientId)
        tvMineValUnified = findViewById(R.id.tvMineValUnified)
        tvMineValHospital = findViewById(R.id.tvMineValHospital)
        tvMineValName = findViewById(R.id.tvMineValName)
        tvMineValGender = findViewById(R.id.tvMineValGender)
        tvMineValAge = findViewById(R.id.tvMineValAge)
        tvMineValTemp = findViewById(R.id.tvMineValTemp)
        tvMineValHr = findViewById(R.id.tvMineValHr)
        tvMineValSpo2 = findViewById(R.id.tvMineValSpo2)
        tvMineValPao2 = findViewById(R.id.tvMineValPao2)
        tvMineValO2hb = findViewById(R.id.tvMineValO2hb)
    }

    private fun applyWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(rootMain) { _, insets ->
            val sys = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            statusBarSpacer.layoutParams = statusBarSpacer.layoutParams.apply { height = sys.top }
            // 底栏为 wrap_content：把系统导航条高度加在 paddingBottom 上，避免固定 56dp 时内容区被挤压裁切
            val extraBottom = (4 * resources.displayMetrics.density).toInt()
            bottomNav.setPadding(0, bottomNav.paddingTop, 0, sys.bottom + extraBottom)
            insets
        }
    }

    private fun setupPatientIdSource() {
        if (isDebugFixedPatientEnabled()) {
            syncPatientIdToLocalProfile(FIXED_PATIENT_ID)
        }
    }

    private fun isDebugFixedPatientEnabled(): Boolean {
        val debuggable =
            (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
        return debuggable && FIXED_PATIENT_ID.isNotBlank()
    }

    private fun currentPatientId(): String {
        if (isDebugFixedPatientEnabled()) return FIXED_PATIENT_ID
        return PatientDataStorage.loadLocalCollectProfile(this).hospitalId.trim()
    }

    private fun startMonitoringFromUi() {
        val id = currentPatientId()
        if (id.isEmpty()) {
            Toast.makeText(this, R.string.toast_collect_patient_data_first, Toast.LENGTH_SHORT).show()
            return
        }
        runWhenBleReady { startMonitoringInternal(fromAuto = false) }
    }

    private fun startMonitoringInternal(fromAuto: Boolean) {
        val id = currentPatientId()
        if (id.isEmpty()) {
            if (!fromAuto) {
                Toast.makeText(this, R.string.toast_collect_patient_data_first, Toast.LENGTH_SHORT).show()
            }
            return
        }

        if (monitoringSessionPatientId != id) {
            monitoringSessionPatientId = id
            hbSeries.clear()
            lacSeries.clear()
            lastHbChartTime = 0L
            lastLacChartTime = 0L
            hbAlarm = false
            updateLegendColors()
        }

        PatientDataStorage.savePatientId(this, id)
        updateHasPatientIdUi()

        isMonitoring = true
        val monitorToken = nextMonitorSessionToken()
        liveMonitorGate.reset()
        handler.removeCallbacks(monitorPollRunnable)
        updateMonitoringUi()

        rebuildMonitorFileSessionForCurrentPatient(id, monitorToken) {
            if (isActiveMonitorSession(id, monitorToken)) {
                startBleMonitoring(id, monitorToken)
            }
        }
    }

    private fun scheduleNextMonitorPoll() {
        handler.removeCallbacks(monitorPollRunnable)
        val periodMs = PatientDataStorage.loadCollectionPeriodSeconds(this) * 1000L
        handler.postDelayed(monitorPollRunnable, periodMs)
    }

    private fun stopMonitoring() {
        isMonitoring = false
        nextMonitorSessionToken()
        handler.removeCallbacks(monitorPollRunnable)
        bleMonitorManager?.stop()
        bleMonitorManager = null
        clearLiveInferenceState()
        flushActiveMonitorSpectrumAsync(clearSession = true)
        liveMonitorGate.reset()
        updateMonitoringUi()
        Toast.makeText(this, R.string.toast_stopped, Toast.LENGTH_SHORT).show()
        startBleStatusMonitor()
    }

    private fun stopMonitoringAfterBleError(message: String) {
        if (!isMonitoring) return
        isMonitoring = false
        nextMonitorSessionToken()
        handler.removeCallbacks(monitorPollRunnable)
        liveMonitorGate.reset()
        bleMonitorManager?.stop(silent = true)
        bleMonitorManager = null
        clearLiveInferenceState()
        flushActiveMonitorSpectrumAsync(clearSession = true)
        updateMonitoringUi()
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        handler.removeCallbacks(bleStatusRetryRunnable)
        handler.postDelayed(bleStatusRetryRunnable, BLE_STATUS_RETRY_MILLIS)
    }

    private fun restartMonitoringForPatientChange() {
        if (!isMonitoring) return
        nextMonitorSessionToken()
        handler.removeCallbacks(monitorPollRunnable)
        bleMonitorManager?.stop(silent = true)
        bleMonitorManager = null
        clearLiveInferenceState()
        flushActiveMonitorSpectrumAsync(clearSession = true)
        liveMonitorGate.reset()
        hbSeries.clear()
        lacSeries.clear()
        lastHbChartTime = 0L
        lastLacChartTime = 0L
        hbAlarm = false
        refreshChart()
        updateLegendColors()
        monitoringSessionPatientId = null
        runWhenBleReady { startMonitoringInternal(fromAuto = true) }
    }

    private fun nextMonitorSessionToken(): Long {
        activeMonitorSessionToken += 1L
        return activeMonitorSessionToken
    }

    private fun nextLiveInferenceRequestId(): Long {
        liveInferenceRequestSequence += 1L
        return liveInferenceRequestSequence
    }

    private fun clearLiveInferenceState() {
        liveInferenceInFlight = false
        pendingLiveInferenceRequestId = null
        pendingInferencePatientId = null
        pendingInferenceSpectrumRange = null
        pendingInferenceMonitorToken = null
    }

    private fun isCurrentLiveInferenceRequest(requestId: Long?): Boolean {
        return requestId != null && pendingLiveInferenceRequestId == requestId
    }

    private fun isActiveMonitorSession(patientId: String, monitorToken: Long): Boolean {
        return isMonitoring &&
            currentPatientId() == patientId &&
            activeMonitorSessionToken == monitorToken
    }

    private fun rebuildMonitorFileSessionForCurrentPatient(
        patientId: String,
        monitorToken: Long,
        onReady: () -> Unit
    ) {
        val profile = PatientDataStorage.loadLocalCollectProfile(this)
        val patientName = profile.name
        val previousSession = activeMonitorFileSession
        activeMonitorFileSession = null
        pendingInferenceSpectrumRange = null
        pendingInferenceMonitorToken = null

        submitMonitorFileIo(
            errorMessage = "患者数据文件初始化失败",
            skipWhenFileWritesDisabled = false
        ) {
            if (monitorFileWritesDisabled) {
                Log.w(TAG, "skip monitor file session init because file writes are disabled")
                postMonitorFileIoFailureToast("患者数据文件初始化失败")
                return@submitMonitorFileIo
            }

            previousSession?.batcher?.flush()

            val patientDir = patientDataFileStore.patientDirectory(patientName, patientId)
            val spectrumFile = patientDataFileStore.spectrumFile(patientDir)
            val initialSampleIndex = MonitorSpectrumFileIndex.nextSampleIndex(spectrumFile)
            val session = MonitorFileSession(
                patientId = patientId,
                monitorToken = monitorToken,
                recordsFile = patientDataFileStore.recordsFile(patientDir),
                batcher = BleSpectrumBatcher(
                    spectrumFile = spectrumFile,
                    sessionId = UUID.randomUUID().toString(),
                    initialSampleIndex = initialSampleIndex
                )
            )
            activeMonitorFileSession = session
            handler.post {
                if (
                    isActiveMonitorSession(patientId, monitorToken) &&
                    activeMonitorFileSession?.monitorToken == monitorToken
                ) {
                    updateMonitoringUi()
                    onReady()
                }
            }
        }
    }

    private fun queueRawSpectrumSamples(
        values: List<BleRawValues>,
        capturedAtMillis: Long,
        expectedPatientId: String,
        expectedMonitorToken: Long
    ) {
        if (values.isEmpty()) return
        submitMonitorFileIo("光谱数据保存失败") {
            val session = activeMonitorFileSession
            if (
                session == null ||
                session.patientId != expectedPatientId ||
                session.monitorToken != expectedMonitorToken
            ) {
                return@submitMonitorFileIo
            }
            session.batcher.addRawValues(values, capturedAtMillis)
        }
    }

    private fun flushActiveMonitorSpectrumAsync(clearSession: Boolean) {
        submitMonitorFileIo(
            errorMessage = "光谱数据写入失败",
            skipWhenFileWritesDisabled = false
        ) {
            val sessionToFlush = activeMonitorFileSession
            try {
                if (!monitorFileWritesDisabled) {
                    sessionToFlush?.batcher?.flush()
                }
            } finally {
                if (clearSession && activeMonitorFileSession === sessionToFlush) {
                    activeMonitorFileSession = null
                }
            }
        }
    }

    private fun submitMonitorFileIo(
        errorMessage: String,
        disableWritesOnFailure: Boolean = true,
        skipWhenFileWritesDisabled: Boolean = true,
        block: () -> Unit
    ) {
        try {
            monitorFileExecutor.execute {
                if (skipWhenFileWritesDisabled && monitorFileWritesDisabled) {
                    return@execute
                }
                try {
                    block()
                } catch (throwable: Throwable) {
                    if (disableWritesOnFailure) {
                        monitorFileWritesDisabled = true
                    }
                    Log.w(TAG, errorMessage, throwable)
                    postMonitorFileIoFailureToast(errorMessage)
                }
            }
        } catch (exception: RejectedExecutionException) {
            Log.w(TAG, "monitor file executor rejected task", exception)
        }
    }

    private fun postMonitorFileIoFailureToast(message: String) {
        val now = System.currentTimeMillis()
        if (!MonitorFileIoFailurePolicy.shouldShowToast(now, lastMonitorFileIoToastAtMillis)) {
            return
        }
        lastMonitorFileIoToastAtMillis = now
        handler.post {
            Toast.makeText(applicationContext, message, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_BLE_PERMISSIONS) return
        blePermissionRequestInFlight = false
        if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            val action = pendingBleReadyAction
            pendingBleReadyAction = null
            action?.invoke() ?: startBleStatusMonitor()
        } else {
            val permanentlyDenied = permissions.any { permission ->
                ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED &&
                    !ActivityCompat.shouldShowRequestPermissionRationale(this, permission)
            }
            pendingBleReadyAction = null
            updateBleConnectionStatus(
                BleConnectionStatus(
                    BleConnectionState.ERROR,
                    BlePermissionPolicy.deniedStatusMessage()
                )
            )
            Toast.makeText(this, BlePermissionPolicy.requestMessage(), Toast.LENGTH_SHORT).show()
            if (permanentlyDenied) {
                showBlePermissionSettingsDialog()
            }
        }
    }

    private fun startBleMonitoring(patientId: String, monitorToken: Long) {
        if (!isActiveMonitorSession(patientId, monitorToken)) return
        handler.removeCallbacks(bleStatusRetryRunnable)
        bleMonitorManager?.stop(silent = true)
        bleMonitorManager = null

        val statusManager = bleStatusManager
        val startAction = BleMonitoringStartPolicy.choose(
            statusSessionReadyToCollect = statusManager?.isReadyToCollect == true
        )
        if (
            startAction == BleMonitoringStartAction.REUSE_CONNECTED_STATUS_SESSION &&
            statusManager != null
        ) {
            bleStatusManager = null
            bleMonitorManager = statusManager
            val reused = statusManager.beginCollecting(
                warmupMillis = LIVE_WARMUP_MILLIS,
                onStatus = { message -> Toast.makeText(this, message, Toast.LENGTH_SHORT).show() },
                onSampleCountChanged = { count, stableElapsedMillis ->
                    if (isActiveMonitorSession(patientId, monitorToken)) {
                        if (liveMonitorGate.markSamples(count, stableElapsedMillis)) {
                            Toast.makeText(this, "稳定采集已达到推理阈值", Toast.LENGTH_SHORT).show()
                            handler.removeCallbacks(monitorPollRunnable)
                            monitorPollRunnable.run()
                        }
                    }
                },
                onError = { message ->
                    if (isActiveMonitorSession(patientId, monitorToken)) {
                        stopMonitoringAfterBleError(message)
                    }
                },
                onRawSamples = { values, capturedAtMillis ->
                    queueRawSpectrumSamples(values, capturedAtMillis, patientId, monitorToken)
                }
            )
            if (reused) return
            statusManager.stop(silent = true)
            bleMonitorManager = null
        }

        bleStatusManager?.stop(silent = true)
        bleStatusManager = null
        bleMonitorManager = BleMonitorManager(
            context = this,
            mode = BleMonitorMode.COLLECTING,
            warmupMillis = LIVE_WARMUP_MILLIS,
            onStatus = { message -> Toast.makeText(this, message, Toast.LENGTH_SHORT).show() },
            onSampleCountChanged = { count, stableElapsedMillis ->
                if (isActiveMonitorSession(patientId, monitorToken)) {
                    if (liveMonitorGate.markSamples(count, stableElapsedMillis)) {
                        Toast.makeText(this, "稳定采集已达到推理阈值", Toast.LENGTH_SHORT).show()
                        handler.removeCallbacks(monitorPollRunnable)
                        monitorPollRunnable.run()
                    }
                }
            },
            onError = { message ->
                if (isActiveMonitorSession(patientId, monitorToken)) {
                    stopMonitoringAfterBleError(message)
                }
            },
            onConnectionStatus = { status -> updateBleConnectionStatus(status) },
            onRawSamples = { values, capturedAtMillis ->
                queueRawSpectrumSamples(values, capturedAtMillis, patientId, monitorToken)
            }
        )
        bleMonitorManager?.start()
    }

    private fun startBleStatusMonitor() {
        handler.removeCallbacks(bleStatusRetryRunnable)
        if (isMonitoring) return
        runWhenBleReady { startBleStatusMonitorAfterReady() }
    }

    private fun startBleStatusMonitorAfterReady() {
        if (isMonitoring) return
        bleStatusManager?.stop(silent = true)
        bleStatusManager = BleMonitorManager(
            context = this,
            mode = BleMonitorMode.STATUS_ONLY,
            onStatus = {},
            onSampleCountChanged = { _, _ -> },
            onError = { message -> handleBleStatusError(message) },
            onConnectionStatus = { status -> updateBleConnectionStatus(status) }
        )
        bleStatusManager?.start()
    }

    private fun handleBleStatusError(message: String) {
        if (isMonitoring) return
        updateBleConnectionStatus(BleConnectionStatus(BleConnectionState.ERROR, message))
        bleStatusManager?.stop(silent = true)
        bleStatusManager = null
        handler.removeCallbacks(bleStatusRetryRunnable)
        handler.postDelayed(bleStatusRetryRunnable, BLE_STATUS_RETRY_MILLIS)
    }

    private fun updateBleConnectionStatus(status: BleConnectionStatus) {
        val colorRes = when (status.state) {
            BleConnectionState.CONNECTED -> R.color.ble_status_connected
            BleConnectionState.ERROR,
            BleConnectionState.DISCONNECTED -> R.color.ble_status_error
            BleConnectionState.SCANNING,
            BleConnectionState.CONNECTING -> R.color.ble_status_pending
            BleConnectionState.IDLE -> R.color.ble_status_idle
        }
        val color = ContextCompat.getColor(this, colorRes)
        tvBleStatusText.text = status.message
        tvBleStatusText.setTextColor(color)
        ViewCompat.setBackgroundTintList(viewBleStatusDot, ColorStateList.valueOf(color))
    }

    private fun hasBlePermissions(): Boolean {
        return BlePermissionPolicy.requiredPermissions().all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun runWhenBleReady(afterReady: () -> Unit) {
        val missingPermissions = BlePermissionPolicy.missingPermissions { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
        when (BleReadyGate.evaluate(missingPermissions, isBleLocationServiceReady())) {
            BleReadiness.READY -> afterReady()
            BleReadiness.NEED_PERMISSIONS -> requestBlePermissions(missingPermissions, afterReady)
            BleReadiness.NEED_LOCATION_SERVICE -> {
                pendingBleReadyAction = afterReady
                updateBleConnectionStatus(
                    BleConnectionStatus(BleConnectionState.ERROR, "请开启定位服务后重试")
                )
                showBleLocationServicesDialog()
            }
        }
    }

    private fun requestBlePermissions(missingPermissions: Array<String>, afterReady: () -> Unit) {
        pendingBleReadyAction = afterReady
        updateBleConnectionStatus(
            BleConnectionStatus(
                BleConnectionState.ERROR,
                BlePermissionPolicy.deniedStatusMessage()
            )
        )
        if (blePermissionRequestInFlight) return
        blePermissionRequestInFlight = true
        ActivityCompat.requestPermissions(this, missingPermissions, REQUEST_BLE_PERMISSIONS)
    }

    private fun isBleLocationServiceReady(): Boolean {
        if (!BlePermissionPolicy.requiresLocationService()) return true
        val locationManager = getSystemService(LocationManager::class.java) ?: return false
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            locationManager.isLocationEnabled
        } else {
            locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER) ||
                locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
        }
    }

    private fun showBlePermissionSettingsDialog() {
        if (blePermissionDialogShowing) return
        blePermissionDialogShowing = true
        MaterialAlertDialogBuilder(this, R.style.ThemeOverlay_BloodCheck_MaterialAlertDialog)
            .setTitle("需要蓝牙权限")
            .setMessage(BlePermissionPolicy.settingsMessage())
            .setPositiveButton("去设置") { _, _ -> openAppSettings() }
            .setNegativeButton("取消", null)
            .setOnDismissListener { blePermissionDialogShowing = false }
            .show()
    }

    private fun showBleLocationServicesDialog() {
        if (bleLocationDialogShowing) return
        bleLocationDialogShowing = true
        MaterialAlertDialogBuilder(this, R.style.ThemeOverlay_BloodCheck_MaterialAlertDialog)
            .setTitle("需要开启定位服务")
            .setMessage("Android 11 及以下系统扫描 BLE 设备需要开启定位服务。开启后返回应用会继续连接蓝牙设备。")
            .setPositiveButton("去开启") { _, _ ->
                startActivity(Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS))
            }
            .setNegativeButton("取消", null)
            .setOnDismissListener { bleLocationDialogShowing = false }
            .show()
    }

    private fun openAppSettings() {
        val intent = Intent(
            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            Uri.fromParts("package", packageName, null)
        )
        startActivity(intent)
    }

    private fun onMonitorPredictionSuccess(response: PredictionResponse, requestId: Long) {
        val spectrumRange = pendingInferenceSpectrumRange
        val monitorToken = pendingInferenceMonitorToken
        val inputId = pendingInferencePatientId ?: currentPatientId()
        if (
            pendingLiveInferenceRequestId != requestId ||
            monitorToken == null ||
            !isActiveMonitorSession(inputId, monitorToken)
        ) {
            Log.w(TAG, "skip stale monitor prediction success for inactive session")
            return
        }
        val qualityDecision = LivePredictionQualityPolicy.validate(response)
        if (!qualityDecision.accepted) {
            bindEmptyMonitoringMetrics()
            Toast.makeText(
                this,
                qualityDecision.message ?: "信号质量不足，请重新采集",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val merged = PatientMonitorMerge.merge(monitorModel, response, inputId)
        monitorModel = merged

        bindPatientHeader(merged)
        bindMonitoringMetrics(merged)
        persistMonitorRecord(merged, response, spectrumRange, monitorToken)
        appendChartPointsFromModel(merged)
        checkHbAlarm()
        refreshChart()
        updateLegendColors()
        updateMonitoringUi()
        if (!currentTabHome) {
            refreshMinePage()
        }
    }

    private fun restoreMonitorStateFromStorage() {
        val id = currentPatientId()
        val localPatch = PatientDataStorage.loadLocalCollectProfile(this).toPatientInfoPatch()
        val latest = if (id.isNotBlank()) {
            PatientDataStorage.pickLatestRecord(PatientDataStorage.getRecords(this, id))
        } else {
            null
        }

        monitorModel = StartupMonitorRestorePolicy.restoreHomeModel(
            currentPatientId = id,
            latestRecord = latest,
            localPatch = localPatch
        )

        val model = monitorModel
        bindPatientHeader(model)
        bindEmptyMonitoringMetrics()
        updateMonitoringUi()
    }

    private fun bindPatientHeader(model: PatientMonitorModel?) {
        val info = model?.patientInfo
        val local = PatientDataStorage.loadLocalCollectProfile(this)

        val pidDisplay = if (model != null) {
            model.patientId
                .ifBlank { info?.get("统一ID").orEmpty() }
                .ifBlank { info?.get("编号").orEmpty() }
                .ifBlank { "--" }
        } else {
            currentPatientId().ifBlank { "--" }
        }

        tvPatientIdRow.text = labelValueSpan(getString(R.string.patient_id_label), pidDisplay)

        val name = local.name.ifBlank { info?.get("姓名").orEmpty() }
        tvPatientName.text = name

        val gender = local.gender.ifBlank { info?.get("性别").orEmpty() }
        when (gender) {
            getString(R.string.gender_male) -> {
                ivPatientGenderIcon.setImageResource(R.drawable.nan)
                ivPatientGenderIcon.visibility = View.VISIBLE
                tvPatientGender.text = getString(R.string.gender_male)
                tvPatientGender.visibility = View.VISIBLE
            }

            getString(R.string.gender_female) -> {
                ivPatientGenderIcon.setImageResource(R.drawable.nv)
                ivPatientGenderIcon.visibility = View.VISIBLE
                tvPatientGender.text = getString(R.string.gender_female)
                tvPatientGender.visibility = View.VISIBLE
            }

            else -> {
                ivPatientGenderIcon.visibility = View.GONE
                tvPatientGender.visibility = View.GONE
            }
        }

        val age = local.age.ifBlank { info?.get("年龄").orEmpty() }
        tvPatientAge.text = if (age.isNotBlank()) "${age}岁" else ""
    }

    private fun bindEmptyMonitoringMetrics() {
        tvDataHb.text = "--"
        tvDataLac.text = "--"
        tvDataPi.text = "--"
        tvDataSpo2.text = "--"
        tvDataPao2.text = ""
        tvDataHr.text = "--"
        tvDataFo2Hb.text = "--"
        tvDataGlucose.text = "--"
        tvDataK.text = "--"
        tvDataNa.text = "--"
    }

    private fun bindMonitoringMetrics(model: PatientMonitorModel) {
        val info = model.patientInfo

        val hbText = model.hemoglobin?.let { h ->
            h.value?.let { v -> "${formatDoubleSmart(v)}${h.unit ?: "g/L"}" }
        } ?: if (!info?.get("血红蛋白").isNullOrBlank()) {
            "${info?.get("血红蛋白")}g/L"
        } else {
            "--"
        }
        tvDataHb.text = hbText
        tvDataHb.setTextColor(
            ContextCompat.getColor(
                this,
                if (hbAlarm) R.color.uni_stop_red else R.color.uni_text_primary
            )
        )

        val lacText = model.lactate?.let { l ->
            l.value?.let { v -> "${formatDoubleSmart(v)}${l.unit ?: "mmol/L"}" }
        } ?: if (!info?.get("乳酸").isNullOrBlank()) {
            "${info?.get("乳酸")}mmol/L"
        } else {
            "--"
        }
        tvDataLac.text = lacText

        val piVal = model.perfusionIndex?.value
        val piText = when {
            piVal != null -> "${piVal.toInt()}%"
            !info?.get("血流灌注指数").isNullOrBlank() -> {
                val v = info?.get("血流灌注指数")?.toDoubleOrNull()
                if (v != null) "${v.toInt()}%" else "--"
            }
            else -> "--"
        }
        tvDataPi.text = piText

        tvDataSpo2.text = if (!info?.get("血氧饱和度").isNullOrBlank()) {
            "${info?.get("血氧饱和度")}%"
        } else {
            "--"
        }

        val pao2Raw = info?.get("氧分压")?.trim().orEmpty()
        tvDataPao2.text = if (pao2Raw.isNotBlank()) {
            "${pao2DisplayInt(pao2Raw)}mmHg"
        } else {
            ""
        }

        tvDataHr.text = if (!info?.get("心率").isNullOrBlank()) {
            "${info?.get("心率")}次/分"
        } else {
            "--"
        }

        val fo2hb = info?.get("氧合血红蛋白分数").orEmpty()
            .ifBlank { info?.get("氧合血红蛋白分数(FO2Hb)").orEmpty() }
        tvDataFo2Hb.text = metricText(fo2hb, "%")
        tvDataGlucose.text = metricText(info?.get("血糖").orEmpty(), "mmol/L")
        tvDataK.text = metricText(info?.get("K+").orEmpty(), "mmol/L")
        tvDataNa.text = metricText(info?.get("Na+").orEmpty(), "mmol/L")
    }

    private fun persistMonitorRecord(
        model: PatientMonitorModel,
        response: PredictionResponse,
        spectrumRange: BleSpectrumRange?,
        monitorToken: Long?
    ) {
        val o = JSONObject()
        o.put("patient_id", model.patientId)
        model.patientInfo?.let { o.put("patient_info", it.toJsonObject()) }
        model.hemoglobin?.let { o.put("hemoglobin", it.toJsonObject()) }
        model.lactate?.let { o.put("lactate", it.toJsonObject()) }
        model.perfusionIndex?.let { o.put("perfusion_index", it.toJsonObject()) }
        o.put("prediction", response.prediction?.toJsonObject())
        val now = System.currentTimeMillis()
        o.put("timestamp", now)
        o.put("dateTime", monitorRecordDateTime(now))
        PatientDataStorage.appendRecord(this, model.patientId, o)

        if (spectrumRange == null) {
            Log.w(TAG, "skip records.csv append because no spectrum range was captured")
            return
        }
        if (monitorToken == null) {
            Log.w(TAG, "skip records.csv append because no monitor session token was captured")
            return
        }

        val filePatientId = currentPatientId()
        submitMonitorFileIo("监测记录文件保存失败") {
            val session = activeMonitorFileSession
            if (
                session == null ||
                session.patientId != filePatientId ||
                session.monitorToken != monitorToken ||
                activeMonitorSessionToken != monitorToken
            ) {
                Log.w(TAG, "skip records.csv append because active file session changed")
                return@submitMonitorFileIo
            }
            session.batcher.flush()
            patientRecordCsvWriter.appendRecord(
                file = session.recordsFile,
                record = o,
                spectrumStartIndex = spectrumRange.startIndex,
                spectrumEndIndex = spectrumRange.endIndex
            )
        }
    }

    private fun monitorRecordDateTime(epochMillis: Long): String {
        return SimpleDateFormat("yyyy/MM/dd HH:mm:ss", Locale.CHINA).format(Date(epochMillis))
    }

    private fun appendChartPointsFromModel(model: PatientMonitorModel) {
        val now = System.currentTimeMillis()
        val hbVal = model.hemoglobin?.value
            ?: model.patientInfo?.get("血红蛋白")?.toDoubleOrNull()
        val lacVal = model.lactate?.value
            ?: model.patientInfo?.get("乳酸")?.toDoubleOrNull()

        if (hbVal != null && now > lastHbChartTime) {
            lastHbChartTime = now
            hbSeries.add(ChartPoint(now, hbVal))
            while (hbSeries.size > 30) hbSeries.removeAt(0)
        }
        if (lacVal != null && now > lastLacChartTime) {
            lastLacChartTime = now
            lacSeries.add(ChartPoint(now, lacVal))
            while (lacSeries.size > 30) lacSeries.removeAt(0)
        }
    }

    private fun currentHbValue(): Double? {
        val m = monitorModel ?: return null
        if (m.hemoglobin?.value != null) return m.hemoglobin.value
        return m.patientInfo?.get("血红蛋白")?.toDoubleOrNull()
    }

    private fun checkHbAlarm() {
        if (hbAlarm) return
        val hb = currentHbValue() ?: return
        val threshold = PatientDataStorage.loadHemoglobinThreshold(this).toDouble()
        if (hb < threshold) {
            hbAlarm = true
        }
    }

    private fun resetHbAlarm() {
        hbAlarm = false
        refreshChart()
        updateLegendColors()
        bindMonitoringMetrics(monitorModel ?: return)
        checkHbAlarm()
        if (hbAlarm) {
            refreshChart()
            updateLegendColors()
            bindMonitoringMetrics(monitorModel ?: return)
        }
        Toast.makeText(this, R.string.toast_alarm_reset, Toast.LENGTH_SHORT).show()
    }

    private fun updateLegendColors() {
        val hbColor = if (hbAlarm) {
            ContextCompat.getColor(this, R.color.design_chart_line_hb_alarm)
        } else {
            ContextCompat.getColor(this, R.color.design_chart_line_hb)
        }
        viewLegendHb.setBackgroundColor(hbColor)
        viewLegendLac.setBackgroundColor(
            ContextCompat.getColor(this, R.color.design_chart_line_lac)
        )
    }

    private fun refreshChart() {
        chartMonitor.setChartData(
            hbSeries.toList(),
            lacSeries.toList(),
            hbAlarm
        )
    }

    private fun updateHasPatientIdUi() {
        //供导出等逻辑使用；界面显隐由 updateMonitoringUi 控制
    }

    private fun hasPatientIdFlag(): Boolean {
        val input = currentPatientId()
        if (input.isNotEmpty()) return true
        val m = monitorModel ?: return false
        if (m.patientId.isNotBlank()) return true
        return !m.patientInfo?.get("统一ID").isNullOrBlank()
    }

    private fun currentPatientNameForFiles(): String {
        return PatientDataStorage.loadLocalCollectProfile(this).name
    }

    private fun currentPatientDatasetDirectory(patientId: String): File {
        val activeSession = activeMonitorFileSession
        val activeDir = activeSession?.recordsFile?.parentFile
        val activeSessionDirectory = activeDir.takeIf {
            activeSession != null && activeSession.patientId == patientId
        }
        return PatientDatasetDirectoryResolver.resolve(
            store = patientDataFileStore,
            patientId = patientId,
            currentPatientName = currentPatientNameForFiles(),
            activeSessionDirectory = activeSessionDirectory
        )
    }

    private fun hasExportablePatientFiles(): Boolean {
        val id = currentPatientId()
        if (id.isEmpty()) return false
        val activeSession = activeMonitorFileSession
        if (activeSession != null && activeSession.patientId == id) return true
        val patientDir = currentPatientDatasetDirectory(id)
        return patientDataFileStore.recordsFile(patientDir).isFile ||
            patientDataFileStore.spectrumFile(patientDir).isFile
    }

    private fun updateMonitoringUi() {
        val hasId = hasPatientIdFlag()
        val hasData = hasExportablePatientFiles()

        if (isMonitoring && hasId) {
            layoutBottomInput.visibility = View.GONE
            layoutMonitorActions.visibility = View.VISIBLE
            btnStopMonitor.visibility = View.VISIBLE
            // 与「停止监测→导出」一致：仅导出按钮相对上一按钮 12dp，本容器顶不额外加距（上一块为图表区）
            applyMonitorActionsOuterPadding(topDp = 12)
        } else {
            layoutBottomInput.visibility = View.VISIBLE
            if (hasId && hasData) {
                layoutMonitorActions.visibility = View.VISIBLE
                btnStopMonitor.visibility = View.GONE
                // 紧接「开始监测」，顶 padding 为 0，间距仅由 btnExport 的 layout_marginTop=12dp 承担
                applyMonitorActionsOuterPadding(topDp = 0)
            } else {
                layoutMonitorActions.visibility = View.GONE
                applyMonitorActionsOuterPadding(topDp = 12)
            }
        }
        tvMonitoringPatientId.visibility = View.GONE
        btnExport.isEnabled = hasId && hasData
    }

    /** 水平 16dp、底 12dp；顶边距按场景变化，使「开始监测→导出」与「停止监测→导出」同为 12dp */
    private fun applyMonitorActionsOuterPadding(topDp: Int) {
        val d = resources.displayMetrics.density
        val h = (16f * d).toInt()
        val top = (topDp * d).toInt()
        val bottom = (12f * d).toInt()
        layoutMonitorActions.setPadding(h, top, h, bottom)
    }

    private fun selectTab(home: Boolean) {
        currentTabHome = home
        scrollHome.visibility = if (home) View.VISIBLE else View.GONE
        scrollMine.visibility = if (home) View.GONE else View.VISIBLE
        tvHeaderTitle.text = getString(
            if (home) R.string.hemoglobin_screen_title else R.string.mine_title
        )
        layoutHeaderActions.visibility = if (home) View.VISIBLE else View.GONE

        val active = ContextCompat.getColor(this, R.color.uni_nav_active)
        val idle = ContextCompat.getColor(this, R.color.uni_text_secondary)
        tvNavHome.setTextColor(if (home) active else idle)
        tvNavMine.setTextColor(if (!home) active else idle)
        // 设计稿 PNG：选中项原色，未选中项套灰色
        if (home) {
            ivNavHome.clearColorFilter()
            ivNavMine.setColorFilter(idle, PorterDuff.Mode.SRC_IN)
        } else {
            ivNavHome.setColorFilter(idle, PorterDuff.Mode.SRC_IN)
            ivNavMine.clearColorFilter()
        }

        if (!home) {
            refreshMinePage()
        }
    }

    private fun refreshMinePage() {
        val pid = currentPatientId()
        if (pid.isEmpty()) {
            layoutMineContent.visibility = View.GONE
            tvMineEmpty.visibility = View.VISIBLE
            return
        }
        val records = PatientDataStorage.getRecords(this, pid)
        if (records.length() == 0) {
            layoutMineContent.visibility = View.VISIBLE
            tvMineEmpty.visibility = View.GONE
            bindMineRowsFromInfo(pid, null)
            return
        }
        val latest = PatientDataStorage.pickLatestRecord(records)
            ?: records.getJSONObject(records.length() - 1)
        val info = latest.optJSONObject("patient_info")?.let { PatientInfo.fromJsonObject(it) }
        val resolvedId = latest.optString("patient_id").ifBlank { pid }
        layoutMineContent.visibility = View.VISIBLE
        tvMineEmpty.visibility = View.GONE
        bindMineRowsFromInfo(resolvedId, info)
    }

    private fun bindMineRowsFromInfo(patientId: String, info: PatientInfo?) {
        val patch = PatientDataStorage.loadLocalCollectProfile(this).toPatientInfoPatch()
        val merged = info?.mergedWith(patch) ?: patch.takeIf { it.values.isNotEmpty() }

        tvMineValPatientId.text = patientId.ifBlank { "--" }
        val unified = merged?.get("统一ID").orEmpty().ifBlank { merged?.get("编号").orEmpty() }
        tvMineValUnified.text = unified.ifBlank { "--" }
        tvMineValHospital.text = merged?.get("住院号") ?: "--"
        tvMineValName.text = merged?.get("姓名") ?: "--"
        tvMineValGender.text = merged?.get("性别") ?: "--"
        val age = merged?.get("年龄").orEmpty()
        tvMineValAge.text = if (age.isNotBlank()) "${age}岁" else "--"
        val temp = merged?.get("体温").orEmpty()
        tvMineValTemp.text = if (temp.isNotBlank()) "${temp}°C" else "--"
        val hr = merged?.get("心率").orEmpty()
        tvMineValHr.text = if (hr.isNotBlank()) "${hr}次/分" else "--"
        val spo2 = merged?.get("血氧饱和度").orEmpty()
        tvMineValSpo2.text = if (spo2.isNotBlank()) "${spo2}%" else "--"
        val pao2 = merged?.get("氧分压").orEmpty()
        tvMineValPao2.text = if (pao2.isNotBlank()) "${pao2DisplayInt(pao2)}mmHg" else ""
        val o2 = merged?.get("氧合血红蛋白分数").orEmpty()
            .ifBlank { merged?.get("氧合血红蛋白分数(FO2Hb)").orEmpty() }
        tvMineValO2hb.text = if (o2.isNotBlank()) "${o2}%" else "--"
    }

    /** 标签 #00D194（design_primary_teal），数值深灰 */
    private fun labelValueSpan(label: String, value: String): CharSequence {
        val green = ContextCompat.getColor(this, R.color.design_primary_teal)
        val dark = ContextCompat.getColor(this, R.color.design_text_primary)
        val text = label + value
        val ss = SpannableString(text)
        ss.setSpan(ForegroundColorSpan(green), 0, label.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        ss.setSpan(ForegroundColorSpan(dark), label.length, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        return ss
    }

    private fun showDataCollectionDialog() {
        val view = LayoutInflater.from(this).inflate(R.layout.dialog_data_collection, null, false)
        val etName = view.findViewById<EditText>(R.id.etCollectName)
        val layoutPickMale = view.findViewById<View>(R.id.layoutPickMale)
        val layoutPickFemale = view.findViewById<View>(R.id.layoutPickFemale)
        val etAge = view.findViewById<EditText>(R.id.etCollectAge)
        val etHospital = view.findViewById<EditText>(R.id.etCollectHospital)
        val etTemp = view.findViewById<EditText>(R.id.etCollectTemp)
        val rbAfNo = view.findViewById<RadioButton>(R.id.rbAfNo)
        val rbAfYes = view.findViewById<RadioButton>(R.id.rbAfYes)
        val btnSubmit = view.findViewById<MaterialButton>(R.id.btnCollectSubmit)

        val saved = PatientDataStorage.loadLocalCollectProfile(this)
        etName.setText(saved.name)
        etAge.setText(saved.age)
        etHospital.setText(saved.hospitalId)
        etTemp.setText(saved.temperature)
        if (saved.atrialFibrillationYes) {
            rbAfYes.isChecked = true
        } else {
            rbAfNo.isChecked = true
        }

        val maleLabel = getString(R.string.gender_male)
        val femaleLabel = getString(R.string.gender_female)
        var selectedGender = when (saved.gender) {
            femaleLabel -> femaleLabel
            maleLabel -> maleLabel
            else -> maleLabel
        }

        fun applyGenderPickUi() {
            val maleSelected = selectedGender == maleLabel
            layoutPickMale.setBackgroundResource(
                if (maleSelected) R.drawable.bg_gender_pick_selected else R.drawable.bg_gender_pick_unselected
            )
            layoutPickFemale.setBackgroundResource(
                if (!maleSelected) R.drawable.bg_gender_pick_selected else R.drawable.bg_gender_pick_unselected
            )
        }

        layoutPickMale.setOnClickListener {
            selectedGender = maleLabel
            applyGenderPickUi()
        }
        layoutPickFemale.setOnClickListener {
            selectedGender = femaleLabel
            applyGenderPickUi()
        }
        applyGenderPickUi()

        val dlg = MaterialAlertDialogBuilder(this, R.style.ThemeOverlay_BloodCheck_MaterialAlertDialog)
            .setView(view)
            .create()
        dlg.setCancelable(true)
        dlg.setCanceledOnTouchOutside(true)

        btnSubmit.setOnClickListener {
            val oldPatientId = currentPatientId()
            val hospitalId = etHospital.text?.toString()?.trim().orEmpty()
            if (hospitalId.isEmpty()) {
                Toast.makeText(this, R.string.toast_collect_patient_data_first, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val patientChangedDuringMonitoring = isMonitoring &&
                oldPatientId.isNotBlank() &&
                oldPatientId != hospitalId
            val profile = LocalCollectProfile(
                name = etName.text?.toString()?.trim().orEmpty(),
                gender = selectedGender,
                age = etAge.text?.toString()?.trim().orEmpty(),
                hospitalId = hospitalId,
                temperature = etTemp.text?.toString()?.trim().orEmpty(),
                atrialFibrillationYes = rbAfYes.isChecked,
            )
            PatientDataStorage.saveLocalCollectProfile(this, profile)

            val patch = profile.toPatientInfoPatch()
            if (patientChangedDuringMonitoring) {
                monitorModel = PatientMonitorModel(
                    patientId = hospitalId,
                    patientInfo = patch.takeIf { it.values.isNotEmpty() },
                    hemoglobin = null,
                    lactate = null,
                    perfusionIndex = null
                )
                bindEmptyMonitoringMetrics()
            } else if (patch.values.isNotEmpty()) {
                val m = monitorModel
                monitorModel = if (m != null) {
                    m.copy(
                        patientId = hospitalId,
                        patientInfo = m.patientInfo?.mergedWith(patch) ?: patch
                    )
                } else {
                    PatientMonitorModel(
                        patientId = hospitalId,
                        patientInfo = patch,
                        hemoglobin = null,
                        lactate = null,
                        perfusionIndex = null
                    )
                }
                monitorModel?.let {
                    bindMonitoringMetrics(it)
                }
            } else if (monitorModel == null) {
                monitorModel = PatientMonitorModel(
                    patientId = hospitalId,
                    patientInfo = null,
                    hemoglobin = null,
                    lactate = null,
                    perfusionIndex = null
                )
                bindEmptyMonitoringMetrics()
            } else {
                val m = monitorModel
                if (m != null && m.patientId.isBlank()) {
                    monitorModel = m.copy(
                        patientId = hospitalId
                    )
                }
            }

            Toast.makeText(this, R.string.data_collect_saved, Toast.LENGTH_SHORT).show()
            bindPatientHeader(monitorModel)
            updateHasPatientIdUi()
            updateMonitoringUi()
            if (!currentTabHome) {
                refreshMinePage()
            }
            if (patientChangedDuringMonitoring) {
                restartMonitoringForPatientChange()
            }
            dlg.dismiss()
        }

        dlg.show()
    }

    private fun showSettingsDialog() {
        val view = LayoutInflater.from(this).inflate(R.layout.dialog_monitor_settings, null, false)
        val etPeriod = view.findViewById<EditText>(R.id.etDialogCollectionPeriod)
        val etThreshold = view.findViewById<EditText>(R.id.etDialogHbThreshold)
        val btnDialogCancel = view.findViewById<MaterialButton>(R.id.btnDialogCancel)
        val btnDialogSave = view.findViewById<MaterialButton>(R.id.btnDialogSave)
        etPeriod.setText(PatientDataStorage.loadCollectionPeriodSeconds(this).toString())
        etThreshold.setText(PatientDataStorage.loadHemoglobinThreshold(this).toString())

        val dlg = MaterialAlertDialogBuilder(this, R.style.ThemeOverlay_BloodCheck_MaterialAlertDialog)
            .setView(view)
            .create()

        btnDialogCancel.setOnClickListener { dlg.dismiss() }
        btnDialogSave.setOnClickListener {
            val period = etPeriod.text?.toString()?.toIntOrNull()
            val threshold = etThreshold.text?.toString()?.toIntOrNull()
            if (period == null || period <= 0) {
                Toast.makeText(this, R.string.settings_invalid_period, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (threshold == null || threshold <= 0) {
                Toast.makeText(this, R.string.settings_invalid_threshold, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            PatientDataStorage.saveSettings(this, period, threshold)
            Toast.makeText(this, R.string.settings_saved, Toast.LENGTH_SHORT).show()
            if (isMonitoring && liveMonitorGate.isOpen) {
                handler.removeCallbacks(monitorPollRunnable)
                monitorPollRunnable.run()
            }
            dlg.dismiss()
        }
        dlg.show()
    }

    private fun exportCurrentPatientData() {
        val id = currentPatientId()
        if (id.isEmpty()) {
            Toast.makeText(this, R.string.toast_export_need_id, Toast.LENGTH_SHORT).show()
            return
        }

        val exportPatientName = currentPatientNameForFiles()
        val outputDir = getExternalFilesDir(android.os.Environment.DIRECTORY_DOWNLOADS)
            ?: filesDir
        val outputZip = File(
            outputDir,
            "患者数据_${PatientDataFileStore.safeName(exportPatientName)}_" +
                "${PatientDataFileStore.safeName(id)}_${System.currentTimeMillis()}.zip"
        )

        submitMonitorFileIo(
            errorMessage = "患者数据导出失败",
            disableWritesOnFailure = false,
            skipWhenFileWritesDisabled = false
        ) {
            val activeSession = activeMonitorFileSession
            if (activeSession != null && activeSession.patientId == id) {
                activeSession.batcher.flush()
            }

            val activeDir = activeSession?.recordsFile?.parentFile
            val activeSessionDirectory = activeDir.takeIf {
                activeSession != null && activeSession.patientId == id
            }
            val patientDir = PatientDatasetDirectoryResolver.resolve(
                store = patientDataFileStore,
                patientId = id,
                currentPatientName = exportPatientName,
                activeSessionDirectory = activeSessionDirectory
            )
            val result = patientZipExporter.zipDataSet(patientDir, outputZip)
            handler.post {
                if (!canShowExportResultUi()) return@post
                if (!result.created) {
                    Toast.makeText(this, R.string.toast_no_patient_files, Toast.LENGTH_SHORT).show()
                    return@post
                }
                showExportSuccessPathDialog(outputZip.absolutePath)
                if (result.missingFiles.isNotEmpty()) {
                    Toast.makeText(this, R.string.export_zip_incomplete, Toast.LENGTH_SHORT).show()
                }
                updateMonitoringUi()
            }
        }
    }

    private fun canShowExportResultUi(): Boolean {
        return !isFinishing && !isDestroyed
    }

    private fun showExportSuccessPathDialog(fullPath: String) {
        val density = resources.displayMetrics.density
        val pad = (20 * density).toInt()
        val secondary = ContextCompat.getColor(this, R.color.design_text_secondary)
        val primary = ContextCompat.getColor(this, R.color.design_text_primary)

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad / 2, pad, 0)
        }
        val hint = TextView(this).apply {
            text = getString(R.string.export_success_dialog_hint)
            textSize = 13f
            setTextColor(secondary)
        }
        val pathTv = TextView(this).apply {
            text = fullPath
            textSize = 14f
            setTextColor(primary)
            setTextIsSelectable(true)
        }
        val scroll = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                (280 * density).toInt()
            )
            addView(pathTv)
        }
        container.addView(hint)
        container.addView(scroll)

        MaterialAlertDialogBuilder(this, R.style.ThemeOverlay_BloodCheck_MaterialAlertDialog)
            .setTitle(R.string.export_success_dialog_title)
            .setView(container)
            .setPositiveButton(R.string.action_ok, null)
            .show()
    }

    private fun formatDoubleSmart(v: Double): String {
        return if (kotlin.math.abs(v - v.toInt()) < 1e-6) {
            v.toInt().toString()
        } else {
            "%.1f".format(v)
        }
    }

    private fun metricText(raw: String, unit: String): String {
        val value = raw.trim()
        if (value.isEmpty()) return "--"
        val number = value.toDoubleOrNull()
        val display = number?.let { formatDoubleSmart(it) } ?: value
        return "$display$unit"
    }

    /** PaO2（氧分压）展示为整数 */
    private fun pao2DisplayInt(raw: String): String {
        val t = raw.trim()
        if (t.isEmpty()) return ""
        val d = t.toDoubleOrNull() ?: return t
        return d.roundToInt().toString()
    }

    private fun syncPatientIdToLocalProfile(patientId: String) {
        val id = patientId.trim()
        if (id.isEmpty()) return
        val saved = PatientDataStorage.loadLocalCollectProfile(this)
        if (saved.hospitalId == id) return
        PatientDataStorage.saveLocalCollectProfile(
            this,
            saved.copy(hospitalId = id)
        )
    }
}

private data class MonitorFileSession(
    val patientId: String,
    val monitorToken: Long,
    val recordsFile: File,
    val batcher: BleSpectrumBatcher
)

internal object PatientDatasetDirectoryResolver {
    fun resolve(
        store: PatientDataFileStore,
        patientId: String,
        currentPatientName: String,
        activeSessionDirectory: File?
    ): File {
        if (activeSessionDirectory != null) return activeSessionDirectory

        val safePatientIdSuffix = "_${PatientDataFileStore.safeName(patientId)}"
        val matching = store.listDataSets()
            .filter { summary -> summary.folderName.endsWith(safePatientIdSuffix) }

        val existing = matching
            .sortedWith(
                compareByDescending<PatientDataSetSummary> { it.isComplete }
                    .thenByDescending { it.lastModifiedMillis }
            )
            .firstOrNull()

        return existing?.dir ?: store.patientDirectory(currentPatientName, patientId)
    }
}

internal object MonitorSpectrumFileIndex {
    fun nextSampleIndex(spectrumFile: File): Long {
        if (!spectrumFile.isFile || spectrumFile.length() == 0L) return 0L
        val tailLine = lastNonEmptyLine(spectrumFile) ?: return 0L
        return firstCsvCell(tailLine)
            ?.toLongOrNull()
            ?.let { it + 1L }
            ?: 0L
    }

    private fun lastNonEmptyLine(file: File): String? {
        RandomAccessFile(file, "r").use { randomAccessFile ->
            var pointer = randomAccessFile.length() - 1L
            val reversedBytes = ArrayList<Byte>()
            var foundContent = false

            while (pointer >= 0L) {
                randomAccessFile.seek(pointer)
                val byte = randomAccessFile.readByte()
                if (byte == '\n'.code.toByte() || byte == '\r'.code.toByte()) {
                    if (foundContent) break
                } else {
                    foundContent = true
                    reversedBytes.add(byte)
                }
                pointer -= 1L
            }

            if (!foundContent) return null
            reversedBytes.reverse()
            return reversedBytes.toByteArray().toString(Charsets.UTF_8)
        }
    }

    private fun firstCsvCell(line: String): String? {
        val trimmed = line.trim().removePrefix("\uFEFF")
        if (trimmed.isEmpty()) return null
        if (!trimmed.startsWith("\"")) {
            return trimmed.substringBefore(",").trim()
        }

        val value = StringBuilder()
        var index = 1
        while (index < trimmed.length) {
            val char = trimmed[index]
            if (char == '"') {
                val next = trimmed.getOrNull(index + 1)
                if (next == '"') {
                    value.append('"')
                    index += 2
                    continue
                }
                return value.toString()
            }
            value.append(char)
            index += 1
        }
        return null
    }
}

internal object MonitorFileIoFailurePolicy {
    private const val TOAST_BACKOFF_MILLIS = 10_000L

    fun shouldShowToast(nowMillis: Long, lastToastMillis: Long): Boolean {
        return lastToastMillis <= 0L || nowMillis - lastToastMillis >= TOAST_BACKOFF_MILLIS
    }
}

private fun JSONObject.toMetricResult(): MetricResult? {
    val value = optDoubleOrNull("value") ?: return null
    return MetricResult(
        value = value,
        unit = optString("unit").takeIf { it.isNotBlank() && it != "null" },
        clinicalInterpretation = optString("clinical_interpretation")
            .takeIf { it.isNotBlank() && it != "null" }
    )
}

private fun JSONObject.toPerfusionIndexResult(): PerfusionIndexResult? {
    val value = optDoubleOrNull("value") ?: return null
    return PerfusionIndexResult(
        value = value,
        classification = optString("classification").takeIf { it.isNotBlank() && it != "null" },
        interpretation = optString("interpretation").takeIf { it.isNotBlank() && it != "null" }
    )
}

private fun JSONObject.optDoubleOrNull(key: String): Double? {
    if (isNull(key) || !has(key)) return null
    return optDouble(key).takeIf { !it.isNaN() }
}
