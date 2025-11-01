using System;
using System.Collections.Generic;
using OpenCvSharp;
using OpenCvSharp.Face;

namespace OpenCvGoruntuIsleme
{
    class Program
    {
        static void Main(string[] args)
        {
            var capture = new VideoCapture(0);
            if (!capture.IsOpened())
            {
                Console.WriteLine("Kamera açılamadı!");
                return;
            }

            var faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            var window = new Window("Yüz Tanıma");
            var image = new Mat();
            var gray = new Mat();

            // ✅ SİNEM'in resimleri (klasörsüz)
            var trainImages = new List<Mat>();
            var labels = new List<int>();

            string[] imagePaths = { "sinem1.jpg", "sinem2.jpg", "sinem3.jpg" };

            foreach (var path in imagePaths)
            {
                var img = Cv2.ImRead(path, ImreadModes.Grayscale);
                trainImages.Add(img);
                labels.Add(1); // Hepsi Sinem
            }

            // ✅ Model eğitimi
            var recognizer = LBPHFaceRecognizer.Create();
            recognizer.Train(trainImages, labels);

            Console.WriteLine("ESC tuşuna basarak çıkabilirsin.");

            while (true)
            {
                capture.Read(image);
                if (image.Empty())
                    continue;

                Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
                var faces = faceCascade.DetectMultiScale(gray, 1.1, 4);

                foreach (var face in faces)
                {
                    var faceROI = new Mat(gray, face);
                    recognizer.Predict(faceROI, out int label, out double confidence);

                    string resultText = (label == 1 && confidence < 60)
                        ? "Bu Sinem :)"
                        : "Sen Sinem değilsin!";

                    Cv2.Rectangle(image, face, Scalar.Green, 2);
                    Cv2.PutText(image, resultText, new Point(face.X, face.Y - 10),
                        HersheyFonts.HersheySimplex, 0.8, Scalar.Yellow, 2);
                }

                window.ShowImage(image);
                if (Cv2.WaitKey(1) == 27) break; // ESC
            }

            capture.Release();
            image.Dispose();
            gray.Dispose();
            window.Close();
        }
    }
}


