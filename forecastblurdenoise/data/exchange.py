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
from forecastblurdenoise.data.traffic import TrafficFormatter

DataFormatter = TrafficFormatter


class ExchangeFormatter(DataFormatter):

    _column_definition = [
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('OT', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('0', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('2', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('3', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('4', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('5', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]
