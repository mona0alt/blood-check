package com.bloodcheck.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class MineFileSelectionUiTest {
    @Test
    fun mineFileActionsDoNotRenderSelectAllOrClearButton() {
        val mainLayout = file("src/main/res/layout/activity_main.xml").readText()

        assertFalse(mainLayout.contains("btnMineFileToggleSelection"))
        assertFalse(mainLayout.contains("file_action_select_all"))
        assertFalse(mainLayout.contains("file_action_clear_selection"))
        assertTrue(mainLayout.contains("btnMineFileRefresh"))
        assertTrue(mainLayout.contains("btnMineFileExport"))
        assertTrue(mainLayout.contains("btnMineFileDelete"))
    }

    @Test
    fun fileSelectionStaysPerRowOnly() {
        val mainActivity = file("src/main/java/com/bloodcheck/app/MainActivity.kt").readText()

        assertFalse(mainActivity.contains("btnMineFileToggleSelection"))
        assertFalse(mainActivity.contains("toggleFileSelection"))
        assertTrue(mainActivity.contains("CheckBox(this@MainActivity)"))
        assertTrue(mainActivity.contains("exportSelectedFileFolder"))
        assertTrue(mainActivity.contains("confirmDeleteSelectedFiles"))
    }

    private fun file(path: String): File =
        File(System.getProperty("user.dir"), path)
}
