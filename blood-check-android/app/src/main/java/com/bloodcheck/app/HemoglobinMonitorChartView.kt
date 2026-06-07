package com.bloodcheck.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.max

data class ChartPoint(val timestamp: Long, val y: Double)

/**
 * 对齐 uniapp [index.vue] 中 canvas 折线图：双 Y 轴、网格、Hb告警红色 / 绿色、Lac 蓝色。
 */
class HemoglobinMonitorChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val hbPoints = mutableListOf<ChartPoint>()
    private val lacPoints = mutableListOf<ChartPoint>()
    private var hbAlarm: Boolean = false

    private var hbMin = 30.0
    private var hbMax = 200.0
    private var lacMin = 0.0
    private var lacMax = 25.0

    private val paintBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val paintGrid = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val paintAxis = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val paintLabel = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val paintLineHb = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val paintLineLac = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    init {
        applyChartColorsFromTheme()
        paintLabel.textSize = sp(12f)
        paintGrid.strokeWidth = dp(1f)
        paintAxis.strokeWidth = dp(1f)
        paintLineHb.strokeWidth = dp(2.5f)
        paintLineLac.strokeWidth = dp(2.5f)
    }

    private fun applyChartColorsFromTheme() {
        val c = context
        paintBg.color = ContextCompat.getColor(c, R.color.design_chart_bg)
        paintGrid.color = ContextCompat.getColor(c, R.color.design_chart_grid)
        paintAxis.color = ContextCompat.getColor(c, R.color.design_chart_axis)
        paintLabel.color = ContextCompat.getColor(c, R.color.design_text_secondary)
        paintLineLac.color = ContextCompat.getColor(c, R.color.design_chart_line_lac)
    }

    fun setChartData(hb: List<ChartPoint>, lac: List<ChartPoint>, alarm: Boolean) {
        hbAlarm = alarm
        hbPoints.clear()
        hbPoints.addAll(hb)
        lacPoints.clear()
        lacPoints.addAll(lac)
        recalculateDynamicScale()
        invalidate()
    }

    private fun recalculateDynamicScale() {
        val now = System.currentTimeMillis()
        val twelveHoursAgo = now - 12 * 60 * 60 * 1000L
        val recentHb = hbPoints.filter { it.timestamp >= twelveHoursAgo }
        val recentLac = lacPoints.filter { it.timestamp >= twelveHoursAgo }

        if (recentLac.isNotEmpty()) {
            val maxLac = recentLac.maxOf { it.y }
            when {
                maxLac < 5 -> {
                    lacMin = 0.0
                    lacMax = 5.0
                }
                maxLac < 10 -> {
                    lacMin = 0.0
                    lacMax = 10.0
                }
                else -> {
                    lacMin = 0.0
                    lacMax = 25.0
                }
            }
        }

        if (recentHb.isNotEmpty()) {
            val maxHb = recentHb.maxOf { it.y }
            hbMin = 30.0
            hbMax = if (maxHb < 100) 100.0 else 200.0
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        val padL = dp(40f)
        val padR = dp(40f)
        val padT = dp(22f)
        val padB = dp(30f)
        val chartW = w - padL - padR
        val chartH = h - padT - padB

        canvas.drawRect(0f, 0f, w, h, paintBg)

        // Y labels left (Hb)
        paintLabel.textAlign = Paint.Align.RIGHT
        for (i in 0..5) {
            val y = padT + (chartH / 5f) * (5 - i)
            val value = hbMin + (hbMax - hbMin) * (i / 5.0)
            canvas.drawText("%.0f".format(value), padL - dp(5f), y + dp(4f), paintLabel)
        }

        // Y labels right (Lac)
        paintLabel.textAlign = Paint.Align.LEFT
        for (i in 0..5) {
            val y = padT + (chartH / 5f) * (5 - i)
            val value = lacMin + (lacMax - lacMin) * (i / 5.0)
            canvas.drawText("%.1f".format(value), w - padR + dp(5f), y + dp(4f), paintLabel)
        }

        // X labels
        paintLabel.textAlign = Paint.Align.CENTER
        if (hbPoints.isNotEmpty()) {
            val timeSpan = (hbPoints.last().timestamp - hbPoints.first().timestamp).coerceAtLeast(1L)
            val timeSpanMinutes = timeSpan / (60 * 1000.0)
            for (i in 0..5) {
                val x = padL + (chartW / 5f) * i
                val timeValue = timeSpanMinutes * (i / 5.0)
                canvas.drawText("%.0f".format(timeValue), x, h - padB + dp(16f), paintLabel)
            }
        } else {
            for (i in 0..5) {
                val x = padL + (chartW / 5f) * i
                canvas.drawText("0", x, h - padB + dp(16f), paintLabel)
            }
        }

        // Grid
        for (i in 0..5) {
            val y = padT + (chartH / 5f) * i
            canvas.drawLine(padL, y, w - padR, y, paintGrid)
        }

        // Y axes
        canvas.drawLine(padL, padT, padL, padT + chartH, paintAxis)
        canvas.drawLine(w - padR, padT, w - padR, padT + chartH, paintAxis)

        // X axis
        canvas.drawLine(padL, padT + chartH, w - padR, padT + chartH, paintAxis)

        // Hb line
        paintLineHb.color = if (hbAlarm) {
            ContextCompat.getColor(context, R.color.design_chart_line_hb_alarm)
        } else {
            ContextCompat.getColor(context, R.color.design_chart_line_hb)
        }
        if (hbPoints.isNotEmpty()) {
            val timeSpan = (hbPoints.last().timestamp - hbPoints.first().timestamp).coerceAtLeast(1L)
            val startTime = hbPoints.first().timestamp
            val path = android.graphics.Path()
            hbPoints.forEachIndexed { index, point ->
                val timeOffset = point.timestamp - startTime
                val x = padL + chartW * (timeOffset.toFloat() / timeSpan.toFloat())
                val y = padT + chartH -
                    ((point.y - hbMin) / max(1e-6, hbMax - hbMin)).toFloat() * chartH
                if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            canvas.drawPath(path, paintLineHb)
        }

        // Lac line
        if (lacPoints.isNotEmpty()) {
            val timeSpan = (lacPoints.last().timestamp - lacPoints.first().timestamp).coerceAtLeast(1L)
            val startTime = lacPoints.first().timestamp
            val path = android.graphics.Path()
            lacPoints.forEachIndexed { index, point ->
                val timeOffset = point.timestamp - startTime
                val x = padL + chartW * (timeOffset.toFloat() / timeSpan.toFloat())
                val y = padT + chartH -
                    ((point.y - lacMin) / max(1e-6, lacMax - lacMin)).toFloat() * chartH
                if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            canvas.drawPath(path, paintLineLac)
        }
    }

    private fun dp(v: Float): Float = v * resources.displayMetrics.density
    private fun sp(v: Float): Float = v * resources.displayMetrics.scaledDensity
}
