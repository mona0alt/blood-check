package com.bloodcheck.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class PatientDataFilesTest {
    @Test
    fun buildsSafePatientFolderNameFromNameAndId() {
        val root = createTempDir(prefix = "patient-files")
        val store = PatientDataFileStore(root)

        val dir = store.patientDirectory(patientName = "张/三 ", patientId = " 0122:004 ")

        assertEquals("张_三_0122_004", dir.name)
        assertEquals(root.absolutePath, dir.parentFile?.absolutePath)
    }

    @Test
    fun listsOnlyPatientDirectoriesWithKnownCsvFiles() {
        val root = createTempDir(prefix = "patient-files")
        val store = PatientDataFileStore(root)
        val patientDir = store.patientDirectory("李四", "p001")
        patientDir.mkdirs()
        File(patientDir, PatientDataFileStore.RECORDS_FILE).writeText("h\n1\n")
        File(patientDir, PatientDataFileStore.SPECTRUM_FILE).writeText("h\n1\n2\n")
        File(root, "loose.csv").writeText("ignore")

        val sets = store.listDataSets()

        assertEquals(1, sets.size)
        assertEquals("李四_p001", sets[0].folderName)
        assertTrue(sets[0].hasRecords)
        assertTrue(sets[0].hasSpectrum)
        assertEquals(2, sets[0].fileCount)
    }

    @Test
    fun exportResolverFindsExistingDatasetForPatientIdAfterNameChanges() {
        val root = createTempDir(prefix = "patient-files")
        val store = PatientDataFileStore(root)
        val oldNameDir = store.patientDirectory("旧姓名", "p001")
        oldNameDir.mkdirs()
        File(oldNameDir, PatientDataFileStore.RECORDS_FILE).writeText("records")

        val resolved = PatientDatasetDirectoryResolver.resolve(
            store = store,
            patientId = "p001",
            currentPatientName = "新姓名",
            activeSessionDirectory = null
        )

        assertEquals(oldNameDir.absolutePath, resolved.absolutePath)
    }

    @Test
    fun exportResolverDoesNotMatchLongerPatientIdWithSameSuffix() {
        val root = createTempDir(prefix = "patient-files")
        val store = PatientDataFileStore(root)
        val wrongDir = store.patientDirectory("患者", "abc_001")
        wrongDir.mkdirs()
        File(wrongDir, PatientDataFileStore.RECORDS_FILE).writeText("wrong")
        val fallback = store.patientDirectory("新姓名", "001")

        val resolved = PatientDatasetDirectoryResolver.resolve(
            store = store,
            patientId = "001",
            currentPatientName = "新姓名",
            activeSessionDirectory = null
        )

        assertEquals(fallback.absolutePath, resolved.absolutePath)
    }

    @Test
    fun exportResolverPrefersActualDataFilesOverNewerMarkerOnlyDirectory() {
        val root = createTempDir(prefix = "patient-files")
        val store = PatientDataFileStore(root)
        val dataDir = store.patientDirectory("旧姓名", "p001")
        dataDir.mkdirs()
        File(dataDir, PatientDataFileStore.RECORDS_FILE).writeText("records")
        val markerDir = store.patientDirectory("新姓名", "p001")
        markerDir.mkdirs()
        File(markerDir, PatientDataFileStore.RECORDS_FILE).mkdir()
        markerDir.setLastModified(dataDir.lastModified() + 10_000L)

        val resolved = PatientDatasetDirectoryResolver.resolve(
            store = store,
            patientId = "p001",
            currentPatientName = "新姓名",
            activeSessionDirectory = null
        )

        assertEquals(dataDir.absolutePath, resolved.absolutePath)
    }

    @Test
    fun csvMarkerDirectoriesDoNotCountAsPatientFiles() {
        val root = createTempDir(prefix = "patient-files")
        val store = PatientDataFileStore(root)
        val patientDir = store.patientDirectory("孙七", "p002")
        patientDir.mkdirs()
        File(patientDir, PatientDataFileStore.RECORDS_FILE).mkdir()
        File(patientDir, PatientDataFileStore.SPECTRUM_FILE).mkdir()

        val sets = store.listDataSets()

        assertEquals(1, sets.size)
        assertFalse(sets[0].hasRecords)
        assertFalse(sets[0].hasSpectrum)
        assertFalse(sets[0].isComplete)
        assertEquals(0, sets[0].fileCount)
    }

    @Test
    fun deletesSelectedPatientDirectoriesButSkipsActivePatient() {
        val root = createTempDir(prefix = "patient-files")
        val store = PatientDataFileStore(root)
        val active = store.patientDirectory("王五", "active")
        val old = store.patientDirectory("赵六", "old")
        active.mkdirs()
        old.mkdirs()
        File(active, PatientDataFileStore.RECORDS_FILE).writeText("active")
        File(old, PatientDataFileStore.RECORDS_FILE).writeText("old")

        val result = store.deleteDataSets(
            folderNames = listOf(active.name, old.name),
            protectedFolderName = active.name
        )

        assertEquals(1, result.deleted)
        assertEquals(1, result.skippedProtected)
        assertTrue(active.exists())
        assertFalse(old.exists())
    }

    @Test
    fun rejectsDeleteFolderNamesThatEscapeRootDirectory() {
        val root = createTempDir(prefix = "patient-files")
        val outside = createTempDir(prefix = "outside-patient-files")
        val store = PatientDataFileStore(root)
        File(outside, PatientDataFileStore.RECORDS_FILE).writeText("outside")

        val result = store.deleteDataSets(
            folderNames = listOf("../${outside.name}"),
            protectedFolderName = null
        )

        assertEquals(1, result.failed)
        assertEquals(0, result.deleted)
        assertTrue(outside.exists())
        assertTrue(File(outside, PatientDataFileStore.RECORDS_FILE).exists())
    }
}
