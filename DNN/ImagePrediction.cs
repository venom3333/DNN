using Microsoft.ML.Data;

using System;
using System.Collections.Generic;
using System.Text;

namespace DNN
{
    /// <summary>
    /// ImagePrediction class holding the score and predicted label metrics.
    /// </summary>
    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] Score { get; set; }

        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
