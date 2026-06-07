package com.bloodcheck.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.lang.reflect.Method
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class BleSpectrumBatcherTest {
    @Test
    fun flushesWhenBatchSizeIsReached() {
        val file = File(tempDir("ble-spectrum-batcher"), "spectrum.csv")
        val batcher = BleSpectrumBatcher(
            spectrumFile = file,
            sessionId = "session-1",
            initialSampleIndex = 0,
            batchSize = 2
        )

        batcher.addRawValues(BleRawValues(c1 = 1, c2 = 2, c3 = 3, red = 4.0, infrared = 5.0), 1000L)

        assertEquals(0L, if (file.exists()) file.length() else 0L)

        batcher.addRawValues(BleRawValues(c1 = 6, c2 = 7, c3 = 8, red = 9.0, infrared = 10.0), 1010L)

        val lines = file.readText(StandardCharsets.UTF_8).removePrefix("\uFEFF").lines().filter { it.isNotEmpty() }
        assertEquals(3, lines.size)
        assertTrueContains(lines[1], "\"0\",\"session-1\"")
        assertTrueContains(lines[2], "\"1\",\"session-1\"")
        assertEquals(1L, batcher.lastFlushedIndex)
        assertEquals(1L, batcher.currentEndIndex)
    }

    @Test
    fun flushesRemainingRowsAndReportsCurrentRange() {
        val file = File(tempDir("ble-spectrum-batcher"), "spectrum.csv")
        val batcher = BleSpectrumBatcher(
            spectrumFile = file,
            sessionId = "session-2",
            initialSampleIndex = 10,
            batchSize = 100
        )

        batcher.addRawValues(BleRawValues(c1 = 1, c2 = 2, c3 = 3, red = 4.0, infrared = 5.0), 1000L)
        batcher.addRawValues(BleRawValues(c1 = 6, c2 = 7, c3 = 8, red = 9.0, infrared = 10.0), 1010L)

        assertEquals(10L, batcher.currentStartIndex)
        assertEquals(11L, batcher.currentEndIndex)

        batcher.flush()

        val lines = file.readText(StandardCharsets.UTF_8).removePrefix("\uFEFF").lines().filter { it.isNotEmpty() }
        assertEquals(3, lines.size)
        assertTrueContains(lines[1], "\"10\",\"session-2\"")
        assertTrueContains(lines[2], "\"11\",\"session-2\"")
        assertEquals(11L, batcher.lastFlushedIndex)
    }

    @Test
    fun listOverloadAssignsSequentialIndexes() {
        val file = File(tempDir("ble-spectrum-batcher"), "spectrum.csv")
        val batcher = BleSpectrumBatcher(
            spectrumFile = file,
            sessionId = "session-list",
            initialSampleIndex = 5,
            batchSize = 3
        )

        batcher.addRawValues(
            listOf(
                BleRawValues(c1 = 1, c2 = 2, c3 = 3, red = 4.0, infrared = 5.0),
                BleRawValues(c1 = 6, c2 = 7, c3 = 8, red = 9.0, infrared = 10.0),
                BleRawValues(c1 = 11, c2 = 12, c3 = 13, red = 14.0, infrared = 15.0)
            ),
            capturedAtMillis = 2000L
        )

        val lines = file.readText(StandardCharsets.UTF_8).removePrefix("\uFEFF").lines().filter { it.isNotEmpty() }
        assertEquals(4, lines.size)
        assertTrueContains(lines[1], "\"5\",\"session-list\"")
        assertTrueContains(lines[2], "\"6\",\"session-list\"")
        assertTrueContains(lines[3], "\"7\",\"session-list\"")
        assertEquals(7L, batcher.lastFlushedIndex)
        assertEquals(7L, batcher.currentEndIndex)
    }

    @Test
    fun currentRangeIsNullBeforeSamplesAndStableAfterSamples() {
        val file = File(tempDir("ble-spectrum-batcher"), "spectrum.csv")
        val batcher = BleSpectrumBatcher(
            spectrumFile = file,
            sessionId = "session-range",
            initialSampleIndex = 20,
            batchSize = 100
        )

        assertEquals(null, batcher.currentRange())

        batcher.addRawValues(
            listOf(
                BleRawValues(c1 = 1, c2 = 2, c3 = 3, red = 4.0, infrared = 5.0),
                BleRawValues(c1 = 6, c2 = 7, c3 = 8, red = 9.0, infrared = 10.0)
            ),
            capturedAtMillis = 3000L
        )

        val range = batcher.currentRange()
        assertEquals(BleSpectrumRange(startIndex = 20, endIndex = 21), range)

        batcher.addRawValues(BleRawValues(c1 = 11, c2 = 12, c3 = 13, red = 14.0, infrared = 15.0), 3010L)

        assertEquals(BleSpectrumRange(startIndex = 20, endIndex = 21), range)
        assertEquals(BleSpectrumRange(startIndex = 20, endIndex = 22), batcher.currentRange())
    }

    @Test
    fun rejectsNonPositiveBatchSize() {
        val file = File(tempDir("ble-spectrum-batcher"), "spectrum.csv")

        try {
            BleSpectrumBatcher(
                spectrumFile = file,
                sessionId = "session-3",
                initialSampleIndex = 0,
                batchSize = 0
            )
            org.junit.Assert.fail("Expected non-positive batchSize to be rejected")
        } catch (expected: IllegalArgumentException) {
            assertEquals("batchSize must be positive", expected.message)
        }
    }

    @Test
    fun concurrentDrainsWriteInDrainOrderAndUpdateLastFlushedAfterWriteCompletes() {
        val file = File(tempDir("ble-spectrum-batcher"), "spectrum.csv")
        val firstWriteStarted = CountDownLatch(1)
        val allowFirstWrite = CountDownLatch(1)
        val secondWriteAttempted = CountDownLatch(1)
        val writes = Collections.synchronizedList(mutableListOf<Long>())
        val threadError = AtomicReference<Throwable?>()
        val batcher = BleSpectrumBatcher(
            spectrumFile = file,
            sessionId = "session-concurrent",
            initialSampleIndex = 0,
            batchSize = 1,
            appendSamples = { _, samples ->
                if (samples.first().sampleIndex == 0L) {
                    firstWriteStarted.countDown()
                    allowFirstWrite.await(5, TimeUnit.SECONDS)
                } else {
                    secondWriteAttempted.countDown()
                }
                writes.add(samples.first().sampleIndex)
            }
        )

        val firstThread = Thread {
            captureThreadError(threadError) {
                batcher.addRawValues(BleRawValues(c1 = 1, c2 = 2, c3 = 3, red = 4.0, infrared = 5.0), 1000L)
            }
        }
        val secondThread = Thread {
            captureThreadError(threadError) {
                batcher.addRawValues(BleRawValues(c1 = 6, c2 = 7, c3 = 8, red = 9.0, infrared = 10.0), 1010L)
            }
        }

        firstThread.start()
        org.junit.Assert.assertTrue(firstWriteStarted.await(5, TimeUnit.SECONDS))

        secondThread.start()

        assertFalse(secondWriteAttempted.await(200, TimeUnit.MILLISECONDS))
        assertEquals(-1L, batcher.lastFlushedIndex)

        allowFirstWrite.countDown()
        firstThread.join(5_000)
        secondThread.join(5_000)

        threadError.get()?.let { throw AssertionError("Worker thread failed", it) }
        assertEquals(listOf(0L, 1L), writes.toList())
        assertEquals(1L, batcher.lastFlushedIndex)
    }

    @Test
    fun interruptedOrderedWriteWaitFailsLaterCallersInsteadOfStrandingThem() {
        val file = File(tempDir("ble-spectrum-batcher"), "spectrum.csv")
        val writes = Collections.synchronizedList(mutableListOf<Long>())
        val batcher = BleSpectrumBatcher(
            spectrumFile = file,
            sessionId = "session-interrupted",
            initialSampleIndex = 0,
            batchSize = 1,
            appendSamples = { _, samples -> writes.add(samples.first().sampleIndex) }
        )
        val writeBatch = privateWriteBatchMethod()
        val waitingBatch = pendingBatch(sequence = 2, sampleIndex = 2)
        val laterBatch = pendingBatch(sequence = 3, sampleIndex = 3)
        val waitingThreadStarted = CountDownLatch(1)
        val waitingThreadError = AtomicReference<Throwable?>()
        val laterThreadError = AtomicReference<Throwable?>()

        val waitingThread = Thread {
            waitingThreadStarted.countDown()
            captureThreadError(waitingThreadError) {
                writeBatch.invoke(batcher, waitingBatch)
            }
        }

        waitingThread.start()
        assertTrue(waitingThreadStarted.await(5, TimeUnit.SECONDS))
        Thread.sleep(100)

        waitingThread.interrupt()
        waitingThread.join(5_000)

        val laterThread = Thread {
            captureThreadError(laterThreadError) {
                writeBatch.invoke(batcher, laterBatch)
            }
        }
        laterThread.start()
        laterThread.join(1_000)

        assertFalse(laterThread.isAlive)
        assertTrue(waitingThreadError.get() is java.lang.reflect.InvocationTargetException)
        assertTrue(laterThreadError.get() is java.lang.reflect.InvocationTargetException)
        assertEquals(emptyList<Long>(), writes.toList())
    }

    private fun assertTrueContains(actual: String, expectedSubstring: String) {
        org.junit.Assert.assertTrue("$actual should contain $expectedSubstring", actual.contains(expectedSubstring))
    }

    private fun captureThreadError(error: AtomicReference<Throwable?>, block: () -> Unit) {
        try {
            block()
        } catch (throwable: Throwable) {
            error.compareAndSet(null, throwable)
        }
    }

    private fun privateWriteBatchMethod(): Method {
        val pendingBatchClass = Class.forName("com.bloodcheck.app.BleSpectrumBatcher\$PendingBatch")
        return BleSpectrumBatcher::class.java.getDeclaredMethod("writeBatch", pendingBatchClass).also {
            it.isAccessible = true
        }
    }

    private fun pendingBatch(sequence: Long, sampleIndex: Long): Any {
        val pendingBatchClass = Class.forName("com.bloodcheck.app.BleSpectrumBatcher\$PendingBatch")
        val constructor = pendingBatchClass.getDeclaredConstructor(Long::class.javaPrimitiveType, List::class.java)
        constructor.isAccessible = true
        return constructor.newInstance(
            sequence,
            listOf(BleRawSample(sampleIndex, "session-interrupted", 1000L, 1, 2, 3, 4.0, 5.0))
        )
    }

    private fun tempDir(prefix: String): File = Files.createTempDirectory(prefix).toFile()
}
