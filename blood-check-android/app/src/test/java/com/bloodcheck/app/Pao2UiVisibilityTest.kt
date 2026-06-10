package com.bloodcheck.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class Pao2UiVisibilityTest {
    @Test
    fun monitorDataLayoutDoesNotRenderPao2ButKeepsHr() {
        val monitorRow = layoutFile("include_monitor_row_pao2_hr.xml").readText()

        assertFalse(monitorRow.contains("data_label_pao2"))
        assertFalse(monitorRow.contains("tvDataPao2"))
        assertTrue(monitorRow.contains("data_label_hr"))
        assertTrue(monitorRow.contains("tvDataHr"))
    }

    @Test
    fun mainLayoutDoesNotRenderMinePao2Row() {
        val mainLayout = layoutFile("activity_main.xml").readText()

        assertFalse(mainLayout.contains("mine_row_pao2"))
        assertFalse(mainLayout.contains("tvMineValPao2"))
        assertTrue(mainLayout.contains("mine_row_spo2"))
        assertTrue(mainLayout.contains("tvMineValSpo2"))
    }

    private fun layoutFile(name: String): File =
        File(System.getProperty("user.dir"), "src/main/res/layout/$name")
}
