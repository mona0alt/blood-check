package com.bloodcheck.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.chaquo.python.Python
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

sealed class PredictUiState {
    data object Idle : PredictUiState()
    data class Loading(val message: String) : PredictUiState()
    data class Success(val response: PredictionResponse) : PredictUiState()
    data class Error(val message: String) : PredictUiState()
}

/** 监测轮询专用状态，不影响旧版单次预测 UI（若仍使用）。 */
sealed class MonitorUiState {
    data object Idle : MonitorUiState()
    data class Loading(val requestId: Long? = null) : MonitorUiState()
    data class Success(val response: PredictionResponse, val requestId: Long? = null) : MonitorUiState()
    data class Error(val message: String, val requestId: Long? = null) : MonitorUiState()
}

class PredictViewModel(application: Application) : AndroidViewModel(application) {

    private val assetCopyHelper = AssetCopyHelper(application.applicationContext)
    private val _uiState = MutableLiveData<PredictUiState>(PredictUiState.Idle)
    val uiState: LiveData<PredictUiState> = _uiState

    private val _monitorState = MutableLiveData<MonitorUiState>(MonitorUiState.Idle)
    val monitorState: LiveData<MonitorUiState> = _monitorState

    fun predict(patientIdInput: String) {
        val patientId = patientIdInput.trim()
        if (patientId.isEmpty()) {
            _uiState.value = PredictUiState.Error("请输入患者ID")
            return
        }

        _uiState.value = PredictUiState.Loading("正在初始化本地数据并预测...")

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val dataRoot = assetCopyHelper.ensureAppDataReady()
                val resultJson = Python.getInstance()
                    .getModule("predict_service")
                    .callAttr("predict_by_patient_id", patientId, dataRoot.absolutePath)
                    .toString()

                val response = PredictionResponse.fromJson(resultJson)
                if (response.success && response.prediction != null) {
                    _uiState.postValue(PredictUiState.Success(response))
                } else {
                    _uiState.postValue(PredictUiState.Error(response.error ?: "模型预测失败"))
                }
            } catch (throwable: Throwable) {
                _uiState.postValue(
                    PredictUiState.Error(
                        throwable.message ?: "预测过程发生未知错误"
                    )
                )
            }
        }
    }

    /**
     * 与「开始预测」相同链路，供首页定时监测调用。
     */
    fun predictForMonitoring(patientIdInput: String) {
        val patientId = patientIdInput.trim()
        if (patientId.isEmpty()) {
            _monitorState.postValue(MonitorUiState.Error("请输入患者ID"))
            return
        }

        _monitorState.postValue(MonitorUiState.Loading())

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val dataRoot = assetCopyHelper.ensureAppDataReady()
                val resultJson = Python.getInstance()
                    .getModule("predict_service")
                    .callAttr("predict_by_patient_id", patientId, dataRoot.absolutePath)
                    .toString()

                val response = PredictionResponse.fromJson(resultJson)
                if (response.success && response.prediction != null) {
                    _monitorState.postValue(MonitorUiState.Success(response))
                } else {
                    _monitorState.postValue(MonitorUiState.Error(response.error ?: "模型预测失败"))
                }
            } catch (throwable: Throwable) {
                _monitorState.postValue(
                    MonitorUiState.Error(
                        throwable.message ?: "预测过程发生未知错误"
                    )
                )
            }
        }
    }

    fun predictLiveForMonitoring(
        patientIdInput: String,
        redValues: List<Double>,
        infraredValues: List<Double>,
        requestId: Long? = null
    ) {
        val patientId = patientIdInput.trim()
        if (patientId.isEmpty()) {
            _monitorState.postValue(MonitorUiState.Error("请输入患者ID", requestId))
            return
        }
        if (redValues.size < 200 || infraredValues.size < 200) {
            _monitorState.postValue(MonitorUiState.Error("实时数据不足，等待采集...", requestId))
            return
        }

        _monitorState.postValue(MonitorUiState.Loading(requestId))

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val dataRoot = assetCopyHelper.ensureAppDataReady()
                val resultJson = Python.getInstance()
                    .getModule("origin_live_inference")
                    .callAttr("predict_live", patientId, redValues, infraredValues, dataRoot.absolutePath)
                    .toString()

                val response = PredictionResponse.fromJson(resultJson)
                if (response.success && response.prediction != null) {
                    _monitorState.postValue(MonitorUiState.Success(response, requestId))
                } else {
                    _monitorState.postValue(MonitorUiState.Error(response.error ?: "实时推理失败", requestId))
                }
            } catch (throwable: Throwable) {
                _monitorState.postValue(
                    MonitorUiState.Error(
                        throwable.message ?: "实时推理过程发生未知错误",
                        requestId
                    )
                )
            }
        }
    }

    fun resetMonitorState() {
        _monitorState.postValue(MonitorUiState.Idle)
    }

    fun resetState() {
        _uiState.value = PredictUiState.Idle
    }
}
