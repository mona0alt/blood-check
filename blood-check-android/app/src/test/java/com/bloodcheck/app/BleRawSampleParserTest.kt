package com.bloodcheck.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class BleRawSampleParserTest {
    @Test
    fun parsesFiveValueArrayIntoRawValues() {
        val parsed = BleRawSampleParser.parse("{array:[87,-51,-2,123640,138385],sum:262059}")

        requireNotNull(parsed)
        assertEquals(87, parsed.c1)
        assertEquals(-51, parsed.c2)
        assertEquals(-2, parsed.c3)
        assertEquals(123640.0, parsed.red, 0.0)
        assertEquals(138385.0, parsed.infrared, 0.0)
    }

    @Test
    fun rejectsArraysThatDoNotHaveFiveValues() {
        assertNull(BleRawSampleParser.parse("{array:[1,2,3,4],sum:10}"))
    }

    @Test
    fun sampleBufferKeepsLatestBoundedPairs() {
        val buffer = BleSignalBuffer(maxPoints = 3)

        buffer.add(BleRawValues(c1 = 0, c2 = 0, c3 = 0, red = 1.0, infrared = 10.0))
        buffer.add(BleRawValues(c1 = 0, c2 = 0, c3 = 0, red = 2.0, infrared = 20.0))
        buffer.add(BleRawValues(c1 = 0, c2 = 0, c3 = 0, red = 3.0, infrared = 30.0))
        buffer.add(BleRawValues(c1 = 0, c2 = 0, c3 = 0, red = 4.0, infrared = 40.0))

        val snapshot = buffer.snapshot()

        assertEquals(listOf(2.0, 3.0, 4.0), snapshot.red)
        assertEquals(listOf(20.0, 30.0, 40.0), snapshot.infrared)
        assertEquals(3, buffer.size)
    }
}
