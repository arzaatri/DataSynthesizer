from typing import Union

import numpy as np
import re
from pandas import Series

from DataSynthesizer.datatypes.AbstractAttribute import AbstractAttribute
from DataSynthesizer.datatypes.utils.DataType import DataType


def pre_process(column: Series):
    if column.size == 0:
        return column
    elif type(column.iloc[0]) is int:
        return column
    elif type(column.iloc[0]) is str:
        return column.map(lambda x: int(x.replace('-', '')))
    else:
        raise Exception('Invalid SocialSecurityNumber.')


def is_ssn(value):
    """
    Test whether a number is a valid social security number. Regex taken from
    https://stackoverflow.com/questions/34523247/social-security-number-validation-that-accepts-dashes-spaces-or-no-spaces
    https://www.codeproject.com/Articles/651609/Validating-Social-Security-Numbers-through-Regular
    With explanations therein

    This function accepts social security numbers in the following forms, where d is a digit:
        ddd-dd-dddd
        ddd dd dddd
        ddddddddd
    And checks adherence to special cases, e.g. that no SSN can start with '666.' For more info, see
    https://www.codeproject.com/Articles/651609/Validating-Social-Security-Numbers-through-Regular
    """
    if type(value) is int:
        value = str(value)
    pattern = re.compile("^((?!219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4})|((?!219 09 9999|078 05 1120)(?!666|000|9\d{2})\d{3} (?!00)\d{2} (?!0{4})\d{4})|((?!219099999|078051120)(?!666|000|9\d{2})\d{3}(?!00)\d{2}(?!0{4})\d{4})$")
    
    if re.search(pattern, value):
        return True
    return False
    
        


class SocialSecurityNumberAttribute(AbstractAttribute):
    """SocialSecurityNumber of format AAA-GG-SSSS."""

    def __init__(self, name: str, is_candidate_key, is_categorical, histogram_size: Union[int, str], data: Series):
        super().__init__(name, is_candidate_key, is_categorical, histogram_size, pre_process(data))
        self.is_numerical = True
        self.data_type = DataType.SOCIAL_SECURITY_NUMBER

    def infer_domain(self, categorical_domain=None, numerical_range=None):
        super().infer_domain(categorical_domain, numerical_range)
        self.min = int(self.min)
        self.max = int(self.max)

    def infer_distribution(self):
        super().infer_distribution()

    def generate_values_as_candidate_key(self, n):
        if n < 1e9:
            values = np.linspace(0, 1e9 - 1, num=n, dtype=int)
            values = np.random.permutation(values)
            values = [str(i).zfill(9) for i in values]
            return ['{}-{}-{}'.format(i[:3], i[3:5], i[5:]) for i in values]
        else:
            raise Exception('The candidate key "{}" cannot generate more than 1e9 distinct values.', self.name)

    def sample_values_from_binning_indices(self, binning_indices):
        return super().sample_binning_indices_in_independent_attribute_mode(binning_indices)
