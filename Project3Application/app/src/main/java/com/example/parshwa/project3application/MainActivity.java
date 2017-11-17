package com.example.parshwa.project3application;

import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.util.SparseArray;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2HSV;



public class MainActivity extends AppCompatActivity {
    //For TextToSpeech Class
    TextToSpeech engine;
    int res;

    static final int REQUEST_IMAGE_CAPTURE = 1;
    ImageView CVImageView;

    //For OCR Code
    TextRecognizer textRecognizer;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button cvButton = (Button) findViewById(R.id.CVButton);
        CVImageView = (ImageView) findViewById(R.id.CVImageView);

        engine = new TextToSpeech(MainActivity.this, new TextToSpeech.OnInitListener(){

            @Override
            public void onInit(int status) {
                if(status == TextToSpeech.SUCCESS) {
                    res = engine.setLanguage(Locale.US);
                } else {
                    Toast.makeText(getApplicationContext(), "Feature is not supported in your device", Toast.LENGTH_SHORT).show();
                }
            }
        });

        //Disable the button if the user has no camera
        if(!hasCamera())
            cvButton.setEnabled(false);


        textRecognizer = new TextRecognizer.Builder(this).build();

        if(!textRecognizer.isOperational()) {
            IntentFilter lowstorageFilter = new IntentFilter(Intent.ACTION_DEVICE_STORAGE_LOW);
            boolean hasLowStorage = registerReceiver(null, lowstorageFilter) != null;

            if (hasLowStorage) {
                Toast.makeText(this,"Low Storage", Toast.LENGTH_LONG).show();
            }
        }
    }



    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status) {
                case LoaderCallbackInterface.SUCCESS:
                    Mat m = new Mat(5, 10, CvType.CV_8UC1, new Scalar(0));
                    break;
                case LoaderCallbackInterface.INIT_FAILED:
                    //Log.i(TAG,"Init Failed");
                    break;
                case LoaderCallbackInterface.INSTALL_CANCELED:
                    //Log.i(TAG,"Install Cancelled");
                    break;
                case LoaderCallbackInterface.INCOMPATIBLE_MANAGER_VERSION:
                    //Log.i(TAG,"Incompatible Version");
                    break;
                case LoaderCallbackInterface.MARKET_ERROR:
                    //Log.i(TAG,"Market Error");
                    break;
                default:
                    //Log.i(TAG,"OpenCV Manager Install");
                    super.onManagerConnected(status);
                    break;
            }
        }
    };



    protected void onDestroy() {
        super.onDestroy();
        if (engine != null) {
            engine.stop();
            engine.shutdown();
        }
    }



    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
    }



    //Check if the user has a camera
    private boolean hasCamera(){
        return getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);
    }



    //Launching the camera
    public void launchCamera(View view){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        //Take a picture and pass results along to onActivityResult
        startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
    }



    //If you want to return the image taken
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if(requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK){
            //Get the photo
            Bundle extras = data.getExtras();
            Bitmap photo = (Bitmap) extras.get("data");



            //Convert image from bitmap into Mat object code
            double w = photo.getWidth();
            double h = photo.getHeight();
            Size size = new Size(w,h);
            try {
                Mat originalImage = new Mat(size, CvType.CV_8U, new Scalar(4));
                Utils.bitmapToMat(photo, originalImage);



                // down-scale and upscale the image to filter out the noise
                /*
                Mat downscaled = new Mat();
                Imgproc.pyrDown(originalImage, downscaled, new Size(originalImage.cols()/2, originalImage.rows()/2));
                Imgproc.pyrUp(downscaled, originalImage, originalImage.size());
                */



                // Code for convert color space into HSV space
                Mat hsvImage = new Mat(originalImage.cols(), originalImage.rows(), CV_8U);
                Imgproc.cvtColor(originalImage, hsvImage, COLOR_RGB2HSV);



                // Code for red color thresholding
                Mat lower_red = new Mat(originalImage.cols(), originalImage.rows(), CV_8U);
                Mat upper_red = new Mat(originalImage.cols(), originalImage.rows(), CV_8U);

                // Red Color Range
                Core.inRange(hsvImage, new Scalar(0, 70, 50), new Scalar(10, 255, 255), lower_red);
                Core.inRange(hsvImage, new Scalar(170, 70, 50), new Scalar(180, 255, 255), upper_red);

                Mat red_img = new Mat(originalImage.cols(), originalImage.rows(), CV_8U);
                Core.bitwise_or(lower_red, upper_red, red_img);
                //Core.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0, red_img);



                // Code for Edge Detection - Canny edge
                Mat cannyEdgeImage = new Mat();
                Imgproc.Canny(red_img, cannyEdgeImage,80, 100);



                // dilate canny output to remove potential holes between edge segments
                //Imgproc.dilate(cannyEdgeImage, cannyEdgeImage, new Mat(), new Point(-1, 1), 1);



                // Code for finding contour
                List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                Mat hierarchy = new Mat();
                Imgproc.findContours(cannyEdgeImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


                int curveNo = -1;
                Mat extractImage = null;
                Mat extractImage1 = null;

                // Loop over all found contours
                for (MatOfPoint cnt : contours) {
                    curveNo++;
                    MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());
                    MatOfPoint2f approxCurve = new MatOfPoint2f();


                    // approximates a polygonal curve with the specified precision
                    Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);

                    int numberVertices = (int) approxCurve.total();
                    double contourArea = Imgproc.contourArea(cnt);

                    // Ignore Small Contours
                    if (Math.abs(contourArea) < 30)
                    {
                        continue;
                    }

                    // Yield Sign Detection - Triangle Detection
                    if(numberVertices == 3) {
                        if (res == TextToSpeech.LANG_NOT_SUPPORTED || res == TextToSpeech.LANG_MISSING_DATA) {
                            Toast.makeText(getApplicationContext(), "Feature is not supported in your device", Toast.LENGTH_SHORT).show();
                        } else {
                            //Bounding Box
                            //Convert back to MatOfPoint
                            MatOfPoint points = new MatOfPoint(approxCurve.toArray());

                            // Get bounding rect of contour
                            Rect rect = Imgproc.boundingRect(points);

                            extractImage1 = new Mat(originalImage, rect);

                            // Convert extractImage mat into ocrImg bitmap
                            Bitmap ocrImg = Bitmap.createBitmap(extractImage1.cols(), extractImage1.rows(), Bitmap.Config.ARGB_8888);
                            Utils.matToBitmap(extractImage1, ocrImg);


                            //OCR Code
                            Frame imageFrame = new Frame.Builder()
                                    .setBitmap(ocrImg)
                                    .build();

                            SparseArray<TextBlock> textBlocks = textRecognizer.detect(imageFrame);

                            String str = "\0";
                            for (int i = 0; i < textBlocks.size(); i++) {
                                TextBlock textBlock = textBlocks.get(textBlocks.keyAt(i));
                                str = str + textBlock.getValue();
                            }

                            // Yield Sign Recognition
                            if (str.equals("\0YIELD")) {
                                engine.speak("Yield sign ahead", TextToSpeech.QUEUE_FLUSH, null, null);
                                Imgproc.rectangle(originalImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 1);
                                Imgproc.drawContours ( originalImage, contours, curveNo, new Scalar(0, 255, 0), 1);
                                Toast.makeText(this, "Yield Sign", Toast.LENGTH_LONG).show();
                            } else {
                                engine.speak("No Yield sign ahead", TextToSpeech.QUEUE_FLUSH, null, null);
                                Toast.makeText(this, "No Yield Sign", Toast.LENGTH_LONG).show();
                            }
                        }
                    }

                    //Octagonal Detection
                    else if (numberVertices >= 8) {

                        List<Double> cos = new ArrayList<>();
                        for (int j = 2; j < numberVertices + 1; j++) {
                            cos.add(
                                    angle(
                                            approxCurve.toArray()[j % numberVertices],
                                            approxCurve.toArray()[j - 2],
                                            approxCurve.toArray()[j - 1]
                                    )
                            );
                        }
                        Collections.sort(cos);

                        double mincos = cos.get(0);
                        double maxcos = cos.get(cos.size() - 1);

                        // Stop Sign Detection
                        if ((numberVertices == 8) && ((mincos >= -0.8) && (maxcos <= -0.0))) {
                            if (res == TextToSpeech.LANG_NOT_SUPPORTED || res == TextToSpeech.LANG_MISSING_DATA) {
                                Toast.makeText(getApplicationContext(), "Feature is not supported in your device", Toast.LENGTH_SHORT).show();
                            } else {
                                //Bounding Box
                                //Convert back to MatOfPoint
                                MatOfPoint points = new MatOfPoint(approxCurve.toArray());

                                // Get bounding rect of contour
                                Rect rect = Imgproc.boundingRect(points);

                                extractImage = new Mat(originalImage, rect);

                                // Convert extractImage mat into ocrImg bitmap
                                Bitmap ocrImg = Bitmap.createBitmap(extractImage.cols(), extractImage.rows(), Bitmap.Config.ARGB_8888);
                                Utils.matToBitmap(extractImage, ocrImg);

                                //OCR Code
                                Frame imageFrame = new Frame.Builder()
                                        .setBitmap(ocrImg)
                                        .build();

                                SparseArray<TextBlock> textBlocks = textRecognizer.detect(imageFrame);

                                String str = "\0";
                                for (int i = 0; i < textBlocks.size(); i++) {
                                    TextBlock textBlock = textBlocks.get(textBlocks.keyAt(i));
                                    str = str + textBlock.getValue();
                                }

                                // Stop Sign Recognition
                                if (str.equals("\0STOP")) {
                                    engine.speak("stop sign ahead", TextToSpeech.QUEUE_FLUSH, null, null);
                                    Imgproc.rectangle(originalImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 1);
                                    Imgproc.drawContours ( originalImage, contours, curveNo, new Scalar(0, 255, 0), 1);
                                    Toast.makeText(this, "Stop Sign", Toast.LENGTH_LONG).show();
                                } else {
                                    engine.speak("No stop sign ahead", TextToSpeech.QUEUE_FLUSH, null, null);
                                    Toast.makeText(this, "No Stop Sign", Toast.LENGTH_LONG).show();
                                }
                            }
                        }
                    }

                    else {
                        if (res == TextToSpeech.LANG_NOT_SUPPORTED || res == TextToSpeech.LANG_MISSING_DATA) {
                            Toast.makeText(getApplicationContext(), "Feature is not supported in your device", Toast.LENGTH_SHORT).show();
                        } else {
                            engine.speak("There is no sign ahead", TextToSpeech.QUEUE_FLUSH, null, null);
                        }
                    }
                }


                Mat result = originalImage.clone();
                // Convert Mat to Bitmap object
                Bitmap bmp = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(result, bmp);


                CVImageView.setImageBitmap(bmp);

            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }

    /**
     * function to find a cosine of angle between vectors
     * from pt0->pt1 and pt0->pt2
     */
    private static double angle(Point pt1, Point pt2, Point pt0)
    {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2)
                / Math.sqrt(
                (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
        );
    }
}
