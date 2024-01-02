# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from forecastblurdenoise.Utils.base import DataTypes, InputTypes
from forecastblurdenoise.data.electricity import ElectricityFormatter

DataFormatter = ElectricityFormatter


class WatershedFormatter(DataFormatter):
    """Defines and formats data_set for the electricity dataset.
        Note that per-entity z-score normalization is used here, and is implemented
        across functions.
        Attributes:
        column_definition: Defines input and data_set type of column used in the
          experiment.
        identifiers: Entity identifiers used in experiments.
        """

    def __init__(self, pred_len):
        super(WatershedFormatter, self).__init__(pred_len)

    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('Conductivity', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('Q', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]
