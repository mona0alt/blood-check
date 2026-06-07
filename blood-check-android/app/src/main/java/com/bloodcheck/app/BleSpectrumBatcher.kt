package com.bloodcheck.app

import androidx.annotation.WorkerThread
import java.io.File

data class BleSpectrumRange(
    val startIndex: Long,
    val endIndex: Long
)

class BleSpectrumBatcher private constructor(
    private val spectrumFile: File,
    private val sessionId: String,
    initialSampleIndex: Long,
    private val batchSize: Int = 100,
    private val appendSamples: (File, List<BleRawSample>) -> Unit,
    @Suppress("UNUSED_PARAMETER") constructorMarker: Unit
) {
    private val lock = Any()
    private val writeLock = Object()
    private val startIndex = initialSampleIndex
    private val pending = mutableListOf<BleRawSample>()
    private var nextSampleIndex = initialSampleIndex
    private var nextBatchSequence = 0L
    private var flushedIndex = initialSampleIndex - 1
    private var nextWriteSequence = 0L
    private var writeFailure: Throwable? = null

    constructor(
        spectrumFile: File,
        sessionId: String,
        initialSampleIndex: Long,
        batchSize: Int = 100,
        writer: SpectrumCsvWriter = SpectrumCsvWriter()
    ) : this(
        spectrumFile = spectrumFile,
        sessionId = sessionId,
        initialSampleIndex = initialSampleIndex,
        batchSize = batchSize,
        appendSamples = writer::appendSamples,
        constructorMarker = Unit
    )

    internal constructor(
        spectrumFile: File,
        sessionId: String,
        initialSampleIndex: Long,
        batchSize: Int = 100,
        appendSamples: (File, List<BleRawSample>) -> Unit
    ) : this(
        spectrumFile = spectrumFile,
        sessionId = sessionId,
        initialSampleIndex = initialSampleIndex,
        batchSize = batchSize,
        appendSamples = appendSamples,
        constructorMarker = Unit
    )

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

        val batchToWrite = synchronized(lock) {
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

            if (pending.size >= batchSize) drainPendingLocked() else null
        }

        writeBatch(batchToWrite)
    }

    @WorkerThread
    fun flush() {
        val batchToWrite = synchronized(lock) {
            if (pending.isEmpty()) null else drainPendingLocked()
        }

        writeBatch(batchToWrite)
    }

    private fun drainPendingLocked(): PendingBatch {
        val samples = pending.toList()
        pending.clear()
        return PendingBatch(sequence = nextBatchSequence++, samples = samples)
    }

    private fun writeBatch(batch: PendingBatch?) {
        if (batch == null) return

        synchronized(writeLock) {
            throwIfWriteFailedLocked()
            while (batch.sequence != nextWriteSequence) {
                try {
                    writeLock.wait()
                } catch (exception: InterruptedException) {
                    Thread.currentThread().interrupt()
                    val failure = IllegalStateException("Interrupted while waiting to write spectrum batch", exception)
                    writeFailure = failure
                    writeLock.notifyAll()
                    throw failure
                }
                throwIfWriteFailedLocked()
            }

            try {
                appendSamples(spectrumFile, batch.samples)
                synchronized(lock) {
                    flushedIndex = batch.samples.last().sampleIndex
                }
            } catch (throwable: Throwable) {
                writeFailure = throwable
                throw throwable
            } finally {
                nextWriteSequence += 1
                writeLock.notifyAll()
            }
        }
    }

    private fun throwIfWriteFailedLocked() {
        writeFailure?.let { failure ->
            throw IllegalStateException("Spectrum batch writer failed", failure)
        }
    }

    private data class PendingBatch(
        val sequence: Long,
        val samples: List<BleRawSample>
    )
}
