package com.bloodcheck.app

import java.io.File

class BleSpectrumBatcher(
    private val spectrumFile: File,
    private val sessionId: String,
    initialSampleIndex: Long,
    private val batchSize: Int = 100,
    private val writer: SpectrumCsvWriter = SpectrumCsvWriter()
) {
    val currentStartIndex: Long = initialSampleIndex

    val currentEndIndex: Long
        get() = nextSampleIndex - 1

    var lastFlushedIndex: Long = initialSampleIndex - 1
        private set

    private val pending = mutableListOf<BleRawSample>()
    private var nextSampleIndex = initialSampleIndex

    init {
        require(batchSize > 0) { "batchSize must be positive" }
    }

    fun addRawValues(values: BleRawValues, capturedAtMillis: Long) {
        pending.add(
            BleRawSample(
                sampleIndex = nextSampleIndex,
                sessionId = sessionId,
                capturedAtMillis = capturedAtMillis,
                c1 = values.c1,
                c2 = values.c2,
                c3 = values.c3,
                red = values.red,
                infrared = values.infrared
            )
        )
        nextSampleIndex += 1

        if (pending.size >= batchSize) {
            flush()
        }
    }

    fun flush() {
        if (pending.isEmpty()) return

        val samples = pending.toList()
        writer.appendSamples(spectrumFile, samples)
        lastFlushedIndex = samples.last().sampleIndex
        pending.clear()
    }
}
