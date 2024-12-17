# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:08:18 2024

@author: Neal
"""

import requests

header = { "accept":"application/json",
            "Content-Type": "application/json"}

test_data = ["苏农银行:苏农银行第六届董事会第十四次会议决议公告",
             "辉丰股份:独立董事候选人声明与承诺-杨兆全",
             "华脉科技:华脉科技2022年年度报告",
             "百济神州:百济神州有限公司章程",
             "莱宝高科:2022年年度权益分派实施公告"]


url = 'http://localhost:9000/predict'

for title in test_data:
    myobj = {'news_title':  title}
    # print(myobj)
    x = requests.post(url, 
                      headers = header,
                      json = myobj)
    print(x.text)



