using Microsoft.ML.Data;

using System;
using System.Collections.Generic;
using System.Text;

namespace DNN
{

    /// <summary>
    /// InMemoryImageData class holding the raw image byte array and label.
    /// </summary>
    public class InMemoryImageData
    {
        [LoadColumn(0)]
        public byte[] Image;

        [LoadColumn(1)]
        [ColumnName("PredictedLabel")]
        public string Label;
    }
}
