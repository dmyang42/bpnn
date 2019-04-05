class InputError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) #初始化父类
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class sample_size_err(InputError):
    def __init__(self):
        super(InputError,self).__init__("Data Size does not fit the initial input number!")
        self.errorinfo="Data Size does not fit the initial input number!"

class weights_size_err(InputError):
    def __init__(self):
        super(InputError,self).__init__("Weights Size does not fit the initial input number!")
        self.errorinfo="Weights Size does not fit the initial input number!"

class thresholds_size_err(InputError):
    def __init__(self):
        super(InputError,self).__init__("Thresholds Size does not fit the initial input number!")
        self.errorinfo="Thresholds Size does not fit the initial input number!"

class sample_label_err(InputError):
    def __init__(self):
        super(InputError,self).__init__("Length of sample does not equal to length of label!")
        self.errorinfo="Length of sample does not equal to length of label!"

class data_type_err(InputError):
    def __init__(self,i):
        super(InputError,self).__init__("Input data should be numeric, but not " + str(type(i)) + "!")
        self.errorinfo="Input data should be numeric, but not " + str(type(i)) + "!"
