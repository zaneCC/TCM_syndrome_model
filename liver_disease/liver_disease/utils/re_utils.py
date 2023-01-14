import re

# 正则过滤，_dict：过滤字典
def rematchFilter(_symp, _dict, match = 0):
    for _pattern in _dict:
        if re.match(_pattern,_symp):
            return match
    return _symp  

# 正则匹配
def rematch(_posture, kindList, nomatch = 0):
    for _pattern in kindList:
        if re.match(kindList[_pattern], _posture):
            return _pattern
    return nomatch

data = '胖大边有齿痕'
KIND_TONGUE_PULSE = {'舌体-胖大':'胖大', '舌体-胖大齿痕':'胖大边有齿痕|胖大，边有齿痕', '舌体-齿痕':'边有齿痕|正常边有齿痕'}
_incaseValue = rematch(data, KIND_TONGUE_PULSE)
print(_incaseValue)