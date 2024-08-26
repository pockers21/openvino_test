#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
from transformers import AutoTokenizer
import argparse
import time

parser = argparse.ArgumentParser(description="加载和使用语言模型")
parser.add_argument("--device", type=str, default="CPU", help="device name")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    parser.add_argument('device')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    system = """\n#角色\n你是一个DSL语法专家，你熟悉antlr设计的一种名为querylang的语言，这种语言用于描述使用各种文件属性来查询操作系统中的文件或文件内容的操作。\n\n#任务\n在下面的任务中将会提供一个自然语言问题，需要将这些自然
    语言转化为满足querylang语法的DSL语句\n\n下面是querylang的语法:\n```antlr\n\ngrammar querylang;\nquery: primary | invalideQuery ;\n\ninvalideQuery: 'ERROR' string;\n\nprimary:\n    binaryExpression ('AND' binaryExpression | 'OR' binaryExpression)*\n    ;\n\nbinaryExpression:\n    '(' primary ')'\n    | searchCondition\n    ;\n\n\nsearchCondition:\n\tdateSearch\n\t| pathSearch\n\t| nameSearch\n\t| sizeSearch\n\t| typeSearch\n\t| durationSearch\n\t| metaSearch\n\t| quantityCondition\n\t| contentSearch;\n\n\ndateSearch: \n    binaryDateSearch ('AND' binaryDateSearch | 'OR' binaryDateSearch)*\n    ;\n\nbinaryDateSearch:\n    '(' dateSearch ')'\n    | dateSearchinfo\n    ;\n\ndateSearchinfo: \n    'DATE' comparison_type absolutedate\n    | 'DATE' comparison_type relativelydate\n    ;\n\nabsolutedate: string;\nrelativelydate: 'CURRENT' '-' string;\n\npathSearch: 'PATH' is_or_not string;\nnameSearch: 'NAME CONTAINS' string;\nsizeSearch:\n\t'SIZE' comparison_type string\n\t| 'SIZE' comparison_type 'FILE_SIZE' filename ;\ntypeSearch: 'TYPE' is_or_not string;\ndurationSearch: 'DURATION' comparison_type string;\nmetaSearch:\n\t'META_TYPE' 'IS' string 'WITH' 'META_VALUE' is_or_not string;\nquantityCondition: 'QUANTITY' '=' NUMBER_VALUE;\ncontentSearch:\n\t'CONTENT CONTAINS' string\n\t| 'CONTENT EQUALS FILE' filename;\n\ncomparison_type: '=' | '<' | '>' | '!=' | '<=' | '>=';\n\nfilename: string;\nstring: STRING_VALUE;\nSTRING_VALUE: '\"' .*? '\"';\n\nis_or_not: 'IS' | 'IS NOT';\nNUMBER_VALUE: [0-9]+;\n\nWS : [ \t\n\r]+ -> skip ;\n\n\n```\n\n#要求：\n1.ROOT是根目录，HOME是家目录或者主目录，Download是下载目录，Pictures是图片目录，以此类推。\n2.请严格按照上面的querylang语法规范来生成DSL代码。\n3.SIZE 可以是各种模糊的表达，例如：超级大，很小，微型等。\n4.>如果用户文件查询请求不是一个文件查询，那么就返回ERROR语句。\n5.DATE可以是模糊的表达，也可以是具体的时间，如果自然问题中是绝对时间信息则不使用CURRENT，如果自然语言问题中是相对时间信息则将相对位置信息修改为以CURRENT为标准的
    绝对时间信息；。\n相对时间信息参考如下：\n\n今天/今天内：(DATE >= CURRENT - \"1 days\") AND (DATE <= \"CURRENT\")\n今天前/早于今天：(DATE < \"CURRENT\")\n昨天：(DATE >= CURRENT - \"2 day\") AND (DATE <= CURRENT - \"1 day\")\n昨天前/早于昨天：(DATE < CURRENT - \"1 day\")\n前天: (DATE >= CURRENT - \"3 day\") AND (DATE <= CURRENT - \"2 day\")\n前天前/早于前天：(DATE < CURRENT - \"2 day\")\n本周/本周内：(DATE >= CURRENT - \"1 week\") AND (DATE <= \"CURRENT\")\n本周前/早于本周：(DATE < CURRENT - \"1 week\")\n上周/上周内：(DATE >= CURRENT - \"2 week\") AND (DATE <= CURRENT - \"1 week\")\n上周前/早于上周: (DATE < CURRENT - \"2 week\")\n上上周内/上上周：(DATE >= CURRENT - \"3 week\") AND (DATE <= CURRENT - \"2 week\")\n上上周前/早于上上周: (DATE < CURRENT - \"3 week\")\n本月/本月内：(DATE >= CURRENT - \"1 month\") AND (DATE <= \"CURRENT\")\n本月前/早于本月: (DATE < CURRENT - \"1 month\")\n上个月：(DATE >= CURRENT - \"2 month\") AND (DATE <= CURRENT - \"1 month\")\n上个月前/早于上个月: (DATE < CURRENT - \"1 month\")\n上上个月：(DATE >= CURRENT - \"3 month\") AND (DATE <= CURRENT - \"2 month\")\n上
    上个月前/早于上上个月: (DATE < CURRENT - \"2 month\")\n今年/今年内：(DATE > CURRENT - \"1 year\") AND (DATE <= \"CURRENT\")\n今年前/早于今年: (DATE < CURRENT - \"1 year\")\n去年：(DATE >= CURRENT - \"2 year\") AND (DATE <= CURRENT - \"1 year\")\n去年前/早于去年: (DATE < CURRENT - \"1 year\")\n前年：(DATE >= CURRENT - \"3 year\") AND (DATE <= CURRENT - \"2 year\")\n前年前/早于前年: (DATE < CURRENT - \"2 year\")\n\n30分钟内/近30分钟：(DATE > CURRENT - \"30 minute\") AND (DATE <= \"CURRENT\")\n16小时内/近16小时：(DATE > CURRENT - \"16 hour\") AND (DATE <= \"CURRENT\")\n3天内/近3天：(DATE > CURRENT - \"3 day\") AND (DATE <= \"CURRENT\")\n2周内/近两周：(DATE > CURRENT - \"2 week\") AND (DATE <= \"CURRENT\")\n3个月内/近3个月：(DATE > CURRENT - \"3 month\") AND (DATE <= \"CURRENT\")\n前2年/2年内/最近两年：(DATE > CURRENT - \"2 year\") AND (DATE <= \"CURRENT\")\n\n30分钟前/30分钟
    之前/早于30分钟之前：(DATE > CURRENT - \"31 minute\") AND (DATE <= CURRENT - \"30 minute\")\n16小时前/16小时之前/早于16小时之前：(DATE > CURRENT - \"17 hour\") AND (DATE <= CURRENT - \"16 hour\")\n3天前/3天之前/早于3天之>前：(DATE > CURRENT - \"4 day\") AND (DATE <= CURRENT - \"3 day\")\n2周前/2周之前/早于两周：(DATE > CURRENT - \"3 week\") AND (DATE <= CURRENT - \"2 week\")   \n3个月前/3个月之前/早于3个月之前：(DATE > CURRENT - \"4 month\") AND (DATE <= CURRENT - \"3 month\")\n2年前：(DATE > CURRENT - \"3 year\") AND (DATE <= CURRENT - \"2 year\")\n\n很多分钟前/几分钟前：(DATE > CURRENT - \"24 min\") AND (DATE <= CURRENT - \"5 min\")\n很多小时前/几小时前
    ：(DATE > CURRENT - \"24 hour\") AND (DATE <= CURRENT - \"5 hour\")\n好几天前/几天前：(DATE > CURRENT - \"24 day\") AND (DATE <= CURRENT - \"5 day\")\n好几周前/(几周前)：(DATE > CURRENT - \"24 week\") AND (DATE <= CURRENT - \"5 week\")\n好几个月前/几个月前：(DATE > CURRENT - \"24 month\") AND (DATE <= CURRENT - \"5 month\")\n很久之前/很多年前/几年前：(DATE > CURRENT - \"24 year\") AND (DATE <= CURRENT - \"5 year\")\n\n最近几分钟：(DATE >= CURRENT - \"5 min\") AND (DATE <= \"CURRENT\")\n近几小时：(DATE >= CURRENT - \"5 hour\") AND (DATE <= \"CURRENT\")\n近几天：(DATE >= CURRENT - \"5 day\") AND (DATE <= \"CURRENT\")\n近几周：(DATE >= CURRENT - \"5 week\") AND (DATE <= \"CURRENT\")\n近几个月：(DATE >= CURRENT - \"5 month\") AND (DATE <= \"CURRENT\")\n近几年：(DATE >= CURRENT - \"5 year\") AND (DATE <= \"CURRENT\")\n\n6、请一定注意，输出结果只能是DSL语句，不可以包含任何其他>内容。\n"""
    prompt = '我想要找那些路径跟“素材.docx”不一样的文件。'
    conv = [{"role": "system", "content": system},
            {"role": "user", "content": prompt},
            ]
    inputs = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")

    device = args.device  # GPU can be used as well
    print(f'use {device}')
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    print(f'inputs:{inputs}')
    inputs_list = inputs[0].tolist()
    decoded_text = tokenizer.decode(inputs_list)

    print("====")
    # 输出解码后的文本
    #print('Decoded text:', decoded_text)

    start_time = time.time()
    res = pipe.generate(decoded_text, config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'cost:{elapsed_time}')
    print(f'res:{res}')


if '__main__' == __name__:
    main()

