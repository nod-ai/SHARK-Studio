from shark.iree_utils._common import check_device_drivers, device_driver_info, IREE_DEVICE_MAP
from parameterized import parameterized

def get_valid_test_params():
    """
    Generate a list of all combinations of available devices and static/dynamic flag.
    """
    device_list = [device for device in IREE_DEVICE_MAP.keys() if not check_device_drivers(device)]
    dynamic_list = (True, False)
    param_list = [(dynamic, device) for dynamic in dynamic_list for device in device_list]
    return param_list

def shark_test_name_func(testcase_func, param_num, param):
    """
    Generate function name string which shows dynamic/static and device name.
    this will be ingested by 'parameterized' package to rename the pytest.
    """
    param_names = []
    for x in param.args:
        if(x == True):
            param_names.append("dynamic")
        elif(x == False):
            param_names.append("static")
        else:
            param_names.append(x)
    return "%s_%s" %(
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param_names)),
    )

# class XfailCases(object):
#     def __init__():
#         device_list = IREE_DEVICE_MAP.keys()
#         self.xfail_cases = {}

#     def insert():

#     def check():
