package com.bloodcheck.app

import androidx.annotation.WorkerThread
import java.io.File

data class BleSpectrumRange(
    val startIndex: Long,
    val endIndex: Long
)

class BleSpectrumBatcher(
    private val spectrumFile: File,
    private val sessionId: String,
    initialSampleIndex: Long,
    private val batchSize: Int = 100,
    private val writer: SpectrumCsvWriter = SpectrumCsvWriter()
) {
    private val lock = Any()
    private val startIndex = initialSampleIndex
    private val pending = mutableListOf<BleRawSample>()
    private var nextSampleIndex = initialSampleIndex
    private var flushedIndex = initialSampleIndex - 1

    val currentStartIndex: Long
        get() = synchronized(lock) { startIndex }

    val currentEndIndex: Long
        get() = synchronized(lock) { nextSampleIndex - 1 }

    val lastFlushedIndex: Long
        get() = synchronized(lock) { flushedIndex }

    init {
        require(batchSize > 0) { "batchSize must be positive" }
    }

    fun currentRange(): BleSpectrumRange? = synchronized(lock) {
        if (nextSampleIndex == startIndex) {
            null
        } else {
            BleSpectrumRange(startIndex = startIndex, endIndex = nextSampleIndex - 1)
        }
    }

    @WorkerThread
    fun addRawValues(values: BleRawValues, capturedAtMillis: Long) {
        addRawValues(listOf(values), capturedAtMillis)
    }

    @WorkerThread
    fun addRawValues(values: List<BleRawValues>, capturedAtMillis: Long) {
        if (values.isEmpty()) return

        val samplesToWrite = synchronized(lock) {
            values.forEach { rawValues ->
                pending.add(
                    BleRawSample(
                        sampleIndex = nextSampleIndex,
                        sessionId = sessionId,
                        capturedAtMillis = capturedAtMillis,
                        c1 = rawValues.c1,
                        c2 = rawValues.c2,
                        c3 = rawValues.c3,
                        red = rawValues.red,
                        infrared = rawValues.infrared
                    )
                )
                nextSampleIndex += 1
            }

            if (pending.size >= batchSize) drainPendingLocked() else emptyList()
        }

        writeSamples(samplesToWrite)
    }

    @WorkerThread
    fun flush() {
        val samplesToWrite = synchronized(lock) {
            if (pending.isEmpty()) emptyList() else drainPendingLocked()
        }

        writeSamples(samplesToWrite)
    }

    private fun drainPendingLocked(): List<BleRawSample> {
        val samples = pending.toList()
        pending.clear()
        return samples
    }

    private fun writeSamples(samples: List<BleRawSample>) {
        if (samples.isEmpty()) return

        writer.appendSamples(spectrumFile, samples)
        synchronized(lock) {
            flushedIndex = maxOf(flushedIndex, samples.last().sampleIndex)
        }
    }
}
