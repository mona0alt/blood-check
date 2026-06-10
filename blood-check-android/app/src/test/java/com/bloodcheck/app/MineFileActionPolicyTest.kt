package com.bloodcheck.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MineFileActionPolicyTest {
    @Test
    fun disablesDeleteAndExportWhenNoFileIsSelected() {
        val state = MineFileActionPolicy.state(selectedCount = 0)

        assertFalse(state.canDelete)
        assertFalse(state.canExport)
    }

    @Test
    fun enablesDeleteAndExportWhenExactlyOneFileIsSelected() {
        val state = MineFileActionPolicy.state(selectedCount = 1)

        assertTrue(state.canDelete)
        assertTrue(state.canExport)
    }

    @Test
    fun enablesDeleteButDisablesExportWhenMultipleFilesAreSelected() {
        val state = MineFileActionPolicy.state(selectedCount = 2)

        assertTrue(state.canDelete)
        assertFalse(state.canExport)
    }
}
