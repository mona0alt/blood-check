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
}
