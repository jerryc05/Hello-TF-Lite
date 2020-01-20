/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.jerryc05.hello_tf_lite

import android.Manifest
import android.app.Fragment
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Camera
import android.hardware.Camera.PreviewCallback
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image.Plane
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.*
import android.util.Size
import android.view.Surface
import android.view.View
import android.view.ViewTreeObserver
import android.view.ViewTreeObserver.OnGlobalLayoutListener
import android.view.WindowManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.appcompat.widget.Toolbar
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetBehavior.BottomSheetCallback
import io.jerryc05.hello_tf_lite.CameraActivity
import io.jerryc05.hello_tf_lite.CameraConnectionFragment.Companion.newInstance
import io.jerryc05.hello_tf_lite.env.ImageUtils.convertYUV420SPToARGB8888
import io.jerryc05.hello_tf_lite.env.ImageUtils.convertYUV420ToARGB8888
import io.jerryc05.hello_tf_lite.env.Logger

abstract class CameraActivity :
    AppCompatActivity(),
    OnImageAvailableListener,
    PreviewCallback,
    CompoundButton.OnCheckedChangeListener,
    View.OnClickListener {

    protected var previewWidth = 0
    protected var previewHeight = 0
    val isDebug = false
    private var handler: Handler? = null
    private var handlerThread: HandlerThread? = null
    private var useCamera2API = false
    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    internal var rgbBytes: IntArray? = null
    protected var luminanceStride = 0
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null

    private var bottomSheetLayout: LinearLayout? = null
    private var gestureLayout: LinearLayout? = null
    private var sheetBehavior: BottomSheetBehavior<*>? = null

    protected var frameValueTextView: TextView? = null
    protected var cropValueTextView: TextView? = null
    protected var inferenceTimeTextView: TextView? = null
    protected var bottomSheetArrowImageView: ImageView? = null
    private var plusImageView: ImageView? = null
    private var minusImageView: ImageView? = null
    private var apiSwitchCompat: SwitchCompat? = null
    private var threadsTextView: TextView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        LOGGER.d("onCreate $this")
        super.onCreate(null)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_camera)
        val toolbar =
            findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar!!.setDisplayShowTitleEnabled(false)
        if (hasPermission()) {
            setFragment()
        } else {
            requestPermission()
        }
        threadsTextView = findViewById(R.id.threads)
        plusImageView = findViewById(R.id.plus)
        minusImageView = findViewById(R.id.minus)
        apiSwitchCompat = findViewById(R.id.api_info_switch)
        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout)
        gestureLayout = findViewById<View>(R.id.gesture_layout) as LinearLayout?
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout!!)
        bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow)
        val vto: ViewTreeObserver = gestureLayout!!.getViewTreeObserver()
        vto.addOnGlobalLayoutListener(
            object : OnGlobalLayoutListener {
                override fun onGlobalLayout() {
                    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
                        gestureLayout!!.getViewTreeObserver().removeGlobalOnLayoutListener(this)
                    } else {
                        gestureLayout!!.getViewTreeObserver().removeOnGlobalLayoutListener(this)
                    }
                    //                int width = bottomSheetLayout.getMeasuredWidth();
                    val height: Int = gestureLayout!!.getMeasuredHeight()
                    sheetBehavior!!.setPeekHeight(height)
                }
            })
        sheetBehavior!!.setHideable(false)
        sheetBehavior!!.setBottomSheetCallback(
            object : BottomSheetCallback() {
                override fun onStateChanged(
                    bottomSheet: View,
                    newState: Int
                ) {
                    when (newState) {
                        BottomSheetBehavior.STATE_HIDDEN -> {
                        }
                        BottomSheetBehavior.STATE_EXPANDED -> {
                            bottomSheetArrowImageView!!.setImageResource(R.drawable.icn_chevron_down)
                        }
                        BottomSheetBehavior.STATE_COLLAPSED -> {
                            bottomSheetArrowImageView!!.setImageResource(R.drawable.icn_chevron_up)
                        }
                        BottomSheetBehavior.STATE_DRAGGING -> {
                        }
                        BottomSheetBehavior.STATE_SETTLING -> bottomSheetArrowImageView!!.setImageResource(
                            R.drawable.icn_chevron_up
                        )
                    }
                }

                override fun onSlide(
                    bottomSheet: View,
                    slideOffset: Float
                ) {
                }
            })
        frameValueTextView = findViewById(R.id.frame_info)
        cropValueTextView = findViewById(R.id.crop_info)
        inferenceTimeTextView = findViewById(R.id.inference_info)
        apiSwitchCompat!!.setOnCheckedChangeListener(this)
        plusImageView!!.setOnClickListener(this)
        minusImageView!!.setOnClickListener(this)
    }

    protected fun getRgbBytes(): IntArray? {
        imageConverter!!.run()
        return rgbBytes
    }

    protected val luminance: ByteArray?
        protected get() = yuvBytes[0]

    /** Callback for android.hardware.Camera API  */
    override fun onPreviewFrame(
        bytes: ByteArray,
        camera: Camera
    ) {
        if (isProcessingFrame) {
            LOGGER.w("Dropping frame!")
            return
        }
        try { // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                val previewSize =
                    camera.parameters.previewSize
                previewHeight = previewSize.height
                previewWidth = previewSize.width
                rgbBytes = IntArray(previewWidth * previewHeight)
                onPreviewSizeChosen(Size(previewSize.width, previewSize.height), 90)
            }
        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            return
        }
        isProcessingFrame = true
        yuvBytes[0] = bytes
        luminanceStride = previewWidth
        imageConverter =
            Runnable { convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes!!) }
        postInferenceCallback = Runnable {
            camera.addCallbackBuffer(bytes)
            isProcessingFrame = false
        }
        processImage()
    }

    /** Callback for Camera2 API  */
    override fun onImageAvailable(reader: ImageReader) { // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return
        }
        if (rgbBytes == null) {
            rgbBytes = IntArray(previewWidth * previewHeight)
        }
        try {
            val image = reader.acquireLatestImage() ?: return
            if (isProcessingFrame) {
                image.close()
                return
            }
            isProcessingFrame = true
            Trace.beginSection("imageAvailable")
            val planes = image.planes
            fillBytes(planes, yuvBytes)
            luminanceStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            imageConverter = object : Runnable {
                override fun run() {
                    convertYUV420ToARGB8888(
                        yuvBytes[0]!!,
                        yuvBytes[1]!!,
                        yuvBytes[2]!!,
                        previewWidth,
                        previewHeight,
                        luminanceStride,
                        uvRowStride,
                        uvPixelStride,
                        rgbBytes!!
                    )
                }
            }
            postInferenceCallback = Runnable {
                image.close()
                isProcessingFrame = false
            }
            processImage()
        } catch (e: Exception) {
            LOGGER.e(e, "Exception!")
            Trace.endSection()
            return
        }
        Trace.endSection()
    }

    @Synchronized
    public override fun onStart() {
        LOGGER.d("onStart $this")
        super.onStart()
    }

    @Synchronized
    public override fun onResume() {
        LOGGER.d("onResume $this")
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

    @Synchronized
    public override fun onPause() {
        LOGGER.d("onPause $this")
        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            LOGGER.e(e, "Exception!")
        }
        super.onPause()
    }

    @Synchronized
    public override fun onStop() {
        LOGGER.d("onStop $this")
        super.onStop()
    }

    @Synchronized
    public override fun onDestroy() {
        LOGGER.d("onDestroy $this")
        super.onDestroy()
    }

    @Synchronized
    protected fun runInBackground(r: Runnable?) {
        if (handler != null) {
            handler!!.post(r)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                setFragment()
            } else {
                requestPermission()
            }
        }
    }

    private fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
                Toast.makeText(
                    this@CameraActivity,
                    "Camera permission is required for this demo",
                    Toast.LENGTH_LONG
                )
                    .show()
            }
            requestPermissions(
                arrayOf(PERMISSION_CAMERA),
                PERMISSIONS_REQUEST
            )
        }
    }

    // Returns true if the device supports the required hardware level, or better.
    private fun isHardwareLevelSupported(
        characteristics: CameraCharacteristics, requiredLevel: Int
    ): Boolean {
        val deviceLevel =
            characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)!!
        return if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            requiredLevel == deviceLevel
        } else requiredLevel <= deviceLevel
        // deviceLevel is not LEGACY, can use numerical sort
    }

    private fun chooseCamera(): String? {
        val manager =
            getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics =
                    manager.getCameraCharacteristics(cameraId)
                // We don't use a front facing camera in this sample.
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue
                }
                val map =
                    characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                        ?: continue
                // Fallback to camera1 API for internal cameras that don't have full support.
// This should help with legacy situations where using the camera2 API causes
// distorted or otherwise broken previews.
                useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL
                        || isHardwareLevelSupported(
                    characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL
                ))
                LOGGER.i("Camera API lv2?: %s", useCamera2API)
                return cameraId
            }
        } catch (e: CameraAccessException) {
            LOGGER.e(e, "Not allowed to access camera")
        }
        return null
    }

    protected fun setFragment() {
        val cameraId = chooseCamera()
        val fragment: Fragment
        if (useCamera2API) {
            val camera2Fragment =
                newInstance(
                    object : CameraConnectionFragment.ConnectionCallback {
                        override fun onPreviewSizeChosen(
                            size: Size?,
                            rotation: Int
                        ) {
                            previewHeight = size!!.height
                            previewWidth = size.width
                            this@CameraActivity.onPreviewSizeChosen(size, rotation)
                        }
                    },
                    this,
                    getLayoutId(),
                    getDesiredPreviewFrameSize()
                )
            camera2Fragment.setCamera(cameraId)
            fragment = camera2Fragment
        } else {
            fragment =
                LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize())
        }
        fragmentManager.beginTransaction().replace(R.id.container, fragment).commit()
    }

    protected fun fillBytes(
        planes: Array<Plane>,
        yuvBytes: Array<ByteArray?>
    ) { // Because of the variable row stride it's not possible to know in
// advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                LOGGER.d(
                    "Initializing buffer %d at size %d",
                    i,
                    buffer.capacity()
                )
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]]
        }
    }

    protected fun readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback!!.run()
        }
    }

    protected val screenOrientation: Int
        protected get() = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }

    override fun onCheckedChanged(
        buttonView: CompoundButton,
        isChecked: Boolean
    ) {
        setUseNNAPI(isChecked)
        if (isChecked) apiSwitchCompat!!.text = "NNAPI" else apiSwitchCompat!!.text = "TFLITE"
    }

    override fun onClick(v: View) {
        if (v.id == R.id.plus) {
            val threads = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            if (numThreads >= 9) return
            numThreads++
            threadsTextView!!.text = numThreads.toString()
            setNumThreads(numThreads)
        } else if (v.id == R.id.minus) {
            val threads = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            if (numThreads == 1) {
                return
            }
            numThreads--
            threadsTextView!!.text = numThreads.toString()
            setNumThreads(numThreads)
        }
    }

    protected fun showFrameInfo(frameInfo: String?) {
        frameValueTextView!!.text = frameInfo
    }

    protected fun showCropInfo(cropInfo: String?) {
        cropValueTextView!!.text = cropInfo
    }

    protected fun showInference(inferenceTime: String?) {
        inferenceTimeTextView!!.text = inferenceTime
    }

    protected abstract fun processImage()
    protected abstract fun onPreviewSizeChosen(size: Size, rotation: Int)
    protected abstract fun getLayoutId(): Int
    protected abstract fun getDesiredPreviewFrameSize(): Size
    protected abstract fun setNumThreads(numThreads: Int)
    protected abstract fun setUseNNAPI(isChecked: Boolean)

    companion object {
        private val LOGGER =
            Logger()
        private const val PERMISSIONS_REQUEST = 1
        private const val PERMISSION_CAMERA = Manifest.permission.CAMERA
        private fun allPermissionsGranted(grantResults: IntArray): Boolean {
            for (result in grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    return false
                }
            }
            return true
        }
    }
}