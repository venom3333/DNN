using Microsoft.ML.Data;

using System;
using System.Collections.Generic;
using System.Text;

namespace DNN
{
    /// <summary>
    /// ImageData class holding the image path and label.
    /// </summary>
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }
}
