#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

# Some libraries and options
import os
from sys import argv

# =============================== MAIN ========================================

if __name__ == "__main__":
    prediction_dir = os.path.join(argv[1], 'res')

    import data_io                       # general purpose input/output functions
    from data_manager import DataManager

    datanames = data_io.inventory_data_nodir(prediction_dir)

    basename = datanames[0]
    D = DataManager(basename, prediction_dir, replace_missing=True, filter_features=True, max_samples=50000, verbose=True)
    print(D)
