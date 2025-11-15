import React, { useState, useRef, useEffect } from 'react';
import { Upload, Settings, Download, ImageIcon, Sliders, Loader } from 'lucide-react';

const ImageToSketchApp = () => {
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [method, setMethod] = useState('canny');
  const [threshold1, setThreshold1] = useState(50);
  const [threshold2, setThreshold2] = useState(150);
  // Kernel size for Gaussian blur must be odd (3, 5, 7, 9, 11)
  const [smoothing, setSmoothing] = useState(5); 
  const [processing, setProcessing] = useState(false);
  
  // Refs cho canvas ẩn để xử lý hình ảnh
  const canvasRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  /**
   * Xử lý tải ảnh lên từ người dùng
   */
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImage(img);
          // Vẽ ảnh gốc lên canvas ẩn
          drawImage(img);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  /**
   * Vẽ ảnh gốc lên canvas ẩn (canvasRef)
   * @param {Image} img - Đối tượng Image đã load
   */
  const drawImage = (img) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Sửa lỗi: Lấy context 2D chính xác
    const ctx = canvas.getContext('2d'); 
    
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };

  /**
   * Chuyển đổi màu RGB sang ảnh xám (Grayscale)
   */
  const toGrayscale = (imageData) => {
    const data = imageData.data;
    const gray = new Uint8ClampedArray(imageData.width * imageData.height);
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Luminosity method (phương pháp độ sáng)
      gray[i / 4] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }
    
    return { data: gray, width: imageData.width, height: imageData.height };
  };

  /**
   * Tạo ma trận kernel Gaussian
   */
  const createGaussianKernel = (size, sigma = 0) => {
    // Sigma mặc định nếu không truyền vào, hoặc tính toán dựa trên kích thước kernel
    if (sigma === 0) sigma = size / 6;
    const kernel = [];
    const half = Math.floor(size / 2);
    let sum = 0;

    for (let y = -half; y <= half; y++) {
      const row = [];
      for (let x = -half; x <= half; x++) {
        const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
        row.push(value);
        sum += value;
      }
      kernel.push(row);
    }
    
    // Normalize kernel
    for(let y = 0; y < size; y++) {
        for(let x = 0; x < size; x++) {
            kernel[y][x] /= sum;
        }
    }
    return kernel;
  };

  /**
   * Bộ lọc Gaussian để làm mịn ảnh xám
   */
  const gaussianBlur = (grayData, kernelSize) => {
    const { data, width, height } = grayData;
    const output = new Uint8ClampedArray(data.length);
    const kernel = createGaussianKernel(kernelSize);
    const half = Math.floor(kernelSize / 2);

    for (let y = half; y < height - half; y++) {
      for (let x = half; x < width - half; x++) {
        let sum = 0;

        for (let ky = -half; ky <= half; ky++) {
          for (let kx = -half; kx <= half; kx++) {
            // Đã loại bỏ clamp/bound check vì ta bắt đầu từ half
            const pixelIndex = (y + ky) * width + (x + kx);
            sum += data[pixelIndex] * kernel[ky + half][kx + half];
          }
        }
        output[y * width + x] = Math.round(sum);
      }
    }

    return { data: output, width, height };
  };

  /**
   * Toán tử Sobel: Tính gradient Magnitude và Direction
   */
  const sobelOperator = (grayData) => {
    const { data, width, height } = grayData;
    const gx = new Float32Array(data.length);
    const gy = new Float32Array(data.length);
    const magnitude = new Uint8ClampedArray(data.length);

    // Sobel kernels
    const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    let maxMag = 0;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sumX = 0, sumY = 0;

        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const pixel = data[(y + ky) * width + (x + kx)];
            sumX += pixel * sobelX[ky + 1][kx + 1];
            sumY += pixel * sobelY[ky + 1][kx + 1];
          }
        }
        
        const idx = y * width + x;
        gx[idx] = sumX;
        gy[idx] = sumY;
        const mag = Math.sqrt(sumX * sumX + sumY * sumY);
        magnitude[idx] = mag;
        if (mag > maxMag) maxMag = mag;
      }
    }

    // Chuẩn hóa Magnitude về 0-255
    for(let i = 0; i < magnitude.length; i++) {
        magnitude[i] = Math.min(255, Math.round(magnitude[i] * 255 / maxMag));
    }

    return { magnitude, gx, gy, width, height };
  };

  /**
   * Non-maximum suppression (cho Canny)
   */
  const nonMaxSuppression = (gradientData) => {
    const { magnitude, gx, gy, width, height } = gradientData;
    const output = new Uint8ClampedArray(magnitude.length);
    output.fill(0); // Khởi tạo với 0 (đen)

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        const angle = Math.atan2(gy[idx], gx[idx]) * 180 / Math.PI;
        let angle_norm = angle < 0 ? angle + 180 : angle;

        let q_idx = -1, r_idx = -1; // Index của 2 điểm lân cận

        // Góc làm tròn về 4 hướng chính (0, 45, 90, 135)
        if ((angle_norm >= 0 && angle_norm < 22.5) || (angle_norm >= 157.5 && angle_norm <= 180)) {
          // 0 độ (ngang)
          q_idx = y * width + (x + 1);
          r_idx = y * width + (x - 1);
        } else if (angle_norm >= 22.5 && angle_norm < 67.5) {
          // 45 độ
          q_idx = (y + 1) * width + (x - 1);
          r_idx = (y - 1) * width + (x + 1);
        } else if (angle_norm >= 67.5 && angle_norm < 112.5) {
          // 90 độ (dọc)
          q_idx = (y + 1) * width + x;
          r_idx = (y - 1) * width + x;
        } else if (angle_norm >= 112.5 && angle_norm < 157.5) {
          // 135 độ
          q_idx = (y - 1) * width + (x - 1);
          r_idx = (y + 1) * width + (x + 1);
        }
        
        // So sánh với 2 điểm lân cận
        if (magnitude[idx] >= magnitude[q_idx] && magnitude[idx] >= magnitude[r_idx]) {
          output[idx] = magnitude[idx];
        }
      }
    }

    return { data: output, width, height };
  };

  /**
   * Hysteresis thresholding (cho Canny)
   */
  const hysteresisThreshold = (nmsData, low, high) => {
    const { data, width, height } = nmsData;
    const output = new Uint8ClampedArray(data.length);
    output.fill(0);
    const strong = 255;
    const weak = 75; // Chỉ dùng để đánh dấu, sẽ được xử lý sau

    for (let i = 0; i < data.length; i++) {
      if (data[i] >= high) {
        output[i] = strong;
      } else if (data[i] >= low) {
        output[i] = weak;
      } else {
        output[i] = 0;
      }
    }

    // Edge tracking bằng cách kết nối các biên 'yếu' với biên 'mạnh'
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        if (output[idx] === weak) {
          let hasStrongNeighbor = false;
          // Kiểm tra 8 lân cận
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              if (output[(y + ky) * width + (x + kx)] === strong) {
                hasStrongNeighbor = true;
                break;
              }
            }
            if (hasStrongNeighbor) break;
          }
          // Nếu có biên mạnh lân cận, thì chuyển biên yếu thành mạnh
          output[idx] = hasStrongNeighbor ? strong : 0;
        }
      }
    }
    
    // Đảm bảo tất cả các điểm 'yếu' còn sót lại (chưa được kết nối) trở thành 0
    for(let i = 0; i < output.length; i++) {
        if(output[i] === weak) output[i] = 0;
    }

    return { data: output, width, height };
  };

  /**
   * Phương pháp Canny Edge Detection (Tối ưu nhất)
   */
  const cannyEdgeDetection = (imageData) => {
    let grayData = toGrayscale(imageData);
    let blurredData = gaussianBlur(grayData, smoothing);
    const gradientData = sobelOperator(blurredData);
    const nmsData = nonMaxSuppression(gradientData);
    return hysteresisThreshold(nmsData, threshold1, threshold2);
  };

  /**
   * Phương pháp Sobel đơn giản
   */
  const simpleSobel = (imageData) => {
    let grayData = toGrayscale(imageData);
    let blurredData = gaussianBlur(grayData, smoothing);
    const { magnitude } = sobelOperator(blurredData);
    
    // Áp dụng ngưỡng đơn
    const output = new Uint8ClampedArray(magnitude.length);
    for (let i = 0; i < magnitude.length; i++) {
      output[i] = magnitude[i] > threshold1 ? 255 : 0;
    }
    
    return { data: output, width: grayData.width, height: grayData.height };
  };

  /**
   * Toán tử Laplacian (Đạo hàm bậc 2)
   */
  const laplacianOperator = (imageData) => {
    let grayData = toGrayscale(imageData);
    let blurredData = gaussianBlur(grayData, smoothing);
    const { data, width, height } = blurredData;
    const output = new Uint8ClampedArray(data.length);
    output.fill(0);

    // Laplacian kernel (tổng của kernel phải là 0)
    const laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]];
    // const laplacian = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]; // Một kernel khác

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sum = 0;

        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            sum += data[(y + ky) * width + (x + kx)] * laplacian[ky + 1][kx + 1];
          }
        }
        
        // Sử dụng Zero-crossing (phát hiện điểm chuyển từ dương sang âm, hoặc ngược lại)
        // Đơn giản hóa: áp dụng ngưỡng lên giá trị tuyệt đối của đạo hàm
        output[y * width + x] = Math.abs(sum) > threshold1 ? 255 : 0;
      }
    }

    return { data: output, width, height };
  };

  /**
   * Hàm chính xử lý chuyển đổi hình ảnh
   */
  const processImage = () => {
    if (!image || processing) return;

    setProcessing(true);
    
    // Sử dụng setTimeout để đảm bảo cập nhật trạng thái UI (Loading) và không chặn luồng chính
    setTimeout(() => {
      try {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Đảm bảo canvas có kích thước chính xác và vẽ lại ảnh (để tránh lỗi khi resize)
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        let edgeData;
        switch(method) {
          case 'canny':
            edgeData = cannyEdgeDetection(imageData);
            break;
          case 'sobel':
            edgeData = simpleSobel(imageData);
            break;
          case 'laplacian':
            edgeData = laplacianOperator(imageData);
            break;
          default:
            edgeData = cannyEdgeDetection(imageData);
        }

        // Vẽ kết quả lên outputCanvas
        const outputCanvas = outputCanvasRef.current;
        outputCanvas.width = edgeData.width;
        outputCanvas.height = edgeData.height;
        const outputCtx = outputCanvas.getContext('2d');
        const outputImageData = outputCtx.createImageData(edgeData.width, edgeData.height);

        // Đảo ngược màu: (255 - value) để có nền trắng, nét đen (hiệu ứng tranh vẽ)
        for (let i = 0; i < edgeData.data.length; i++) {
          const value = 255 - edgeData.data[i];
          const dataIdx = i * 4;
          outputImageData.data[dataIdx] = value;
          outputImageData.data[dataIdx + 1] = value;
          outputImageData.data[dataIdx + 2] = value;
          outputImageData.data[dataIdx + 3] = 255; // Alpha channel
        }

        outputCtx.putImageData(outputImageData, 0, 0);
        setProcessedImage(outputCanvas.toDataURL('image/png'));
      } catch (error) {
        console.error("Lỗi xử lý hình ảnh:", error);
      } finally {
        setProcessing(false);
      }
    }, 50);
  };

  /**
   * Tải ảnh đã xử lý xuống
   */
  const downloadImage = () => {
    if (!processedImage) return;
    
    const link = document.createElement('a');
    link.download = `sketch_image_${method}.png`;
    link.href = processedImage;
    link.click();
  };

  // Tự động xử lý lại ảnh mỗi khi các tham số thay đổi
  useEffect(() => {
    if (image) {
      processImage();
    }
  }, [image, method, threshold1, threshold2, smoothing]);


  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 sm:p-6 font-sans">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-3xl shadow-2xl p-6 sm:p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-extrabold text-gray-800 mb-2 tracking-tight">
              🎨 Chuyển Ảnh Thành Tranh Vẽ
            </h1>
          </div>

          {/* Upload Section */}
          <div className="mb-8">
            <div className="border-4 border-dashed border-indigo-400 rounded-xl p-8 text-center bg-indigo-50 hover:bg-indigo-100 transition-all cursor-pointer shadow-lg"
                 onClick={() => fileInputRef.current?.click()}>
              <Upload className="mx-auto mb-4 text-indigo-600" size={48} />
              <p className="text-xl font-bold text-gray-700 mb-2">
                {image ? "Tải ảnh khác lên" : "Tải ảnh lên"}
              </p>
              <p className="text-sm text-gray-500">
                Nhấp để chọn ảnh (JPEG, PNG, v.v.)
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </div>
          </div>

          {/* Settings Panel */}
          {image && (
            <div className="mb-8 bg-gradient-to-r from-purple-50 to-pink-50 rounded-2xl p-6 shadow-xl border-t-4 border-purple-500">
              <div className="flex items-center mb-6">
                <Settings className="mr-2 text-purple-600" size={24} />
                <h2 className="text-2xl font-bold text-gray-800">Cài đặt Thuật toán</h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                
                {/* Phương pháp */}
                <div className="col-span-1 md:col-span-2 lg:col-span-1">
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Phương pháp phát hiện biên
                  </label>
                  <select
                    value={method}
                    onChange={(e) => setMethod(e.target.value)}
                    className="w-full p-3 border-2 border-purple-300 rounded-lg shadow-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition"
                  >
                    <option value="canny">Canny (Tối ưu - Khuyên dùng)</option>
                    <option value="sobel">Sobel (Đơn giản)</option>
                    <option value="laplacian">Laplacian (Đạo hàm bậc 2)</option>
                  </select>
                </div>

                {/* Độ làm mịn */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Độ làm mịn (Gaussian Blur): <span className="font-bold text-purple-600">{smoothing}</span>
                  </label>
                  <input
                    type="range"
                    min="3"
                    max="11"
                    step="2"
                    value={smoothing}
                    onChange={(e) => setSmoothing(parseInt(e.target.value))}
                    className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer transition-all"
                  />
                </div>

                {/* Ngưỡng thấp (T1) */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Ngưỡng thấp (T1/Threshold): <span className="font-bold text-purple-600">{threshold1}</span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="255"
                    value={threshold1}
                    onChange={(e) => setThreshold1(parseInt(e.target.value))}
                    className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer transition-all"
                  />
                </div>

                {/* Ngưỡng cao (T2) - Chỉ cho Canny */}
                {method === 'canny' && (
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Ngưỡng cao (T2): <span className="font-bold text-purple-600">{threshold2}</span>
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="255"
                      value={threshold2}
                      onChange={(e) => setThreshold2(parseInt(e.target.value))}
                      className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer transition-all"
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Image Display */}
          {image && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              
              {/* Ảnh Gốc */}
              <div className="bg-gray-100 rounded-xl p-5 shadow-inner">
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center border-b pb-2">
                  <ImageIcon className="mr-2 text-indigo-500" size={24} />
                  Ảnh gốc (Original)
                </h3>
                <div className="border-4 border-gray-300 rounded-lg overflow-hidden max-h-[80vh] flex justify-center items-center bg-white">
                  <img src={image.src} alt="Ảnh gốc" className="w-full h-auto object-contain" />
                </div>
              </div>

              {/* Tranh Vẽ */}
              <div className="bg-gray-100 rounded-xl p-5 shadow-inner">
                <div className="flex justify-between items-center mb-4 border-b pb-2">
                  <h3 className="text-xl font-bold text-gray-800 flex items-center">
                    <Sliders className="mr-2 text-pink-500" size={24} />
                    Tranh vẽ (Sketch)
                  </h3>
                  {processedImage && (
                    <button
                      onClick={downloadImage}
                      className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-xl transition-all font-semibold shadow-md hover:shadow-lg transform hover:scale-105"
                      disabled={processing}
                    >
                      <Download size={18} />
                      Tải xuống
                    </button>
                  )}
                </div>
                
                <div className="border-4 border-gray-300 rounded-lg overflow-hidden max-h-[80vh] flex justify-center items-center bg-white aspect-square">
                  {processing ? (
                    <div className="flex flex-col items-center justify-center h-64 text-indigo-600">
                      <Loader className="animate-spin h-12 w-12 mb-3" />
                      <p className="font-medium">Đang xử lý...</p>
                      <p className="text-sm text-gray-500 mt-1">Vui lòng đợi giây lát</p>
                    </div>
                  ) : processedImage ? (
                    // Hiển thị ảnh đã xử lý
                    <img src={processedImage} alt="Ảnh đã xử lý" className="w-full h-auto object-contain" />
                  ) : (
                    <div className="flex items-center justify-center h-64 text-gray-400">
                      Chưa có kết quả xử lý
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          
          {/* Hidden canvases - Phải giữ lại để xử lý hình ảnh */}
          <canvas ref={canvasRef} className="hidden" />
          <canvas ref={outputCanvasRef} className="hidden" />

        </div>
      </div>
      
    </div>
  );
};

export default ImageToSketchApp;
