
# 필요 패키지 추가
from generalmodule import combinedata, mergedata, insertdata, preprocessdata, encodingdata
from splitmodule1 import get_pattern, make_data
from splitmodule2 import splitdata
import time
import datetime
import pickle
import glob

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFRegressor, XGBRFClassifier, XGBClassifier
import matplotlib.font_manager as fm

# 한글깨짐 방지코드 
font_location = '/home/sagemaker-user/gsc/NanumGothic.ttf'
fm.fontManager.addfont(font_location)
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rc('axes', unicode_minus=False)

# 웹 페이지 기본 설정
# page title: 데이터 분석 및 모델링 대시보드
st.set_page_config(
    page_title="Priority opening fixed equipment", # page 타이틀
    page_icon="🧊", # page 아이콘
    layout="wide", # wide, centered
    initial_sidebar_state="auto", # 사이드 바 초기 상태
    menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': None,
        'About': '2023 GS CDS Class',

    }
)


import streamlit as st
import fitz
from PIL import Image
import io
import os

def pdf_to_images(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]

        # Convert the page to a pixmap
        pixmap = page.get_pixmap()
        
#         img = img.resize((target_width, target_height))

        # Convert the pixmap to bytes using Pillow
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        images.append(img_bytes)

    pdf_document.close()
    return images

def front_page():
    # st.title("PDF Image Viewer")
    cols = st.columns((1, 1, 1))
    with cols[1]:
        st.write('Apps for')
    cols = st.columns((1, 4, 1))
    with cols[1]:
        st.subheader('**Considering Priority Opening Fixed Equipment**')
    cols = st.columns((1, 3, 1))
    with cols[1]:

        st.markdown("")
        st.divider()
        st.write(' 1. Preprocessing')
        # st.divider()
        st.write(' 2. EDA')
        # st.divider()
        st.write(' 3. Modeling')
        # st.divider()
        st.write(' 4. Model Serving')
        # st.divider()
        st.write(' 5. Result')
        st.divider()
        
        
    # 현재 작업 디렉토리를 기준으로 상대 경로 사용
    script_directory = os.getcwd()
    pdf_path = os.path.join(script_directory, "GSCaltex_Signature_slogan.pdf")

    images = pdf_to_images(pdf_path)
    st.markdown("")
    st.markdown("")
    cols = st.columns((3, 3, 1))
    with cols[1]:
        st.write("By")
    # Display images in Streamlit
    for i, image in enumerate(images):
        st.image(image, use_column_width=True, width=100)

        
def file_merging():
    # 파일 업로더 위젯 추가 (1번쨰)
    file = st.file_uploader("Please upload PSM files", type=['csv', 'xls', 'xlsx'], accept_multiple_files=True)

    if file is not None:
        # 새 파일이 업로드되면 기존 상태 초기화
        st.session_state['preprocessing_state'] = {}
        st.session_state['eda_state'] = {}
        st.session_state['modeling_state'] = {}
        st.session_state['serving_state'] = {}
        
        # 새로 업로드된 파일 저장
        st.session_state['preprocessing_state']['current_file'] = file
    
    # 새로 업로드한 파일을 df로 로드
    if 'current_data' in st.session_state['preprocessing_state'] and len(st.session_state['preprocessing_state']['current_file']) > 0:
        if st.session_state['preprocessing_state']['current_data']:
            # 첫 번째 파일을 선택 (리스트에서 파일을 선택하는 방식으로 변경)
            selected_file = st.session_state['preprocessing_state']['current_data'][0]

            # 파일 이름 출력
            if hasattr(selected_file, 'name'):
                st.write(f"Current File: {selected_file.name}")
            else:
                st.write("Current File: Does not have 'name' attribute")

            # 선택된 파일을 df로 로드
            st.session_state['preprocessing_state']['current_data'] = load_file(selected_file)

    # # 새로 로드한 df 저장
    # if 'current_data' in st.session_state['preprocessing_state']:
        # st.dataframe(st.session_state['preprocessing_state']['current_data'])
        
    if st.session_state['preprocessing_state']['current_file'] is not None and len(st.session_state['preprocessing_state']['current_file']) > 0:
        # 업로드된 파일들을 합쳐서 DataFrame으로 가져옴
        
        combined_data = combinedata(st.session_state['preprocessing_state']['current_file'])

        # # 결과를 표시
        # st.dataframe(combined_data)
        
        # splitdata 함수 호출
        df_split = splitdata(combined_data)
        st.session_state['preprocessing_state']['current_file'] = df_split
        st.session_state['preprocessing_state']['df_split'] = df_split
        # 결과를 표시
        # st.dataframe(df_split)
        
# files 변수를 초기화
    files = []

    if 'current_file' in st.session_state['preprocessing_state'] and len(st.session_state['preprocessing_state']['current_file']) > 0:
        files = st.file_uploader("Please upload EPS files", type=['csv', 'xls', 'xlsx'], accept_multiple_files=True, key="file_uploader")

        # 업로드된 파일이 있는지 확인
        if files is not None:
            # 새로 업로드된 파일 저장
            st.session_state['preprocessing_state']['current_file'] = files

            # 업로드된 각 파일을 읽어 DataFrame으로 변환
            dfs_uploaded = [pd.read_excel(file, header=None) for file in files]
            if len(dfs_uploaded) > 0:
                df_eps = dfs_uploaded[0]

                st.session_state['preprocessing_state']['current_data'] = df_eps

                # 새로 로드한 df 저장
                if 'current_data' in st.session_state['preprocessing_state']:

                    df_eps = df_eps.rename(columns={0: '장치번호', 1: '설비등급', 2:'설비유형',
                                                              6:'부식율', 7:'잔여수명'})
                    df_eps[15] = np.nan
                    df_eps[16] = np.nan
                    df_eps['우선개방여부'] = '해당없음'   
                    df_eps.columns = df_eps.columns.astype(str)
                    st.session_state['preprocessing_state']['eps_data'] = df_eps
                    # st.dataframe(st.session_state['preprocessing_state']['eps_data'])
                    # st.dataframe(st.session_state['preprocessing_state']['df_split'])
                    # df_a와 df_uploaded를 merge
                    df_merged = mergedata(st.session_state['preprocessing_state']['eps_data'], st.session_state['preprocessing_state']['df_split'])
                    st.session_state['preprocessing_state']['eps_data'] = df_merged
                    # st.dataframe(st.session_state['preprocessing_state']['eps_data'])
                    

    if 'eps_data' in st.session_state['preprocessing_state'] and len(st.session_state['preprocessing_state']['eps_data']) > 0:
        filess = st.file_uploader("Please upload EQL file", type=['csv', 'xls', 'xlsx'])

        # 업로드된 파일이 있는지 확인
        if filess is not None:
            # 새로 업로드된 파일 저장
            st.session_state['preprocessing_state']['current_equlist'] = filess
            
        # 새로 업로드한 장치List 파일을 df로 로드
        if 'current_equlist' in st.session_state['preprocessing_state']:
            # st.write(f"Current File_equipment list: {st.session_state['preprocessing_state']['current_equlist'].name}")
            equlist = pd.read_excel(st.session_state['preprocessing_state']['current_equlist'], skiprows=1, engine='openpyxl')
            st.session_state['preprocessing_state']['equ_data'] = equlist
            df_final = insertdata(st.session_state['preprocessing_state']['equ_data'])
            st.session_state['preprocessing_state']['final_data'] = df_final
            st.dataframe(st.session_state['preprocessing_state']['final_data'])
                    
                
def file_preprocessing():
    if 'final_data' in st.session_state['preprocessing_state'] and len(st.session_state['preprocessing_state']['final_data']) > 0:     
        df_pre = preprocessdata(st.session_state['preprocessing_state']['final_data'])
        st.session_state['preprocessing_state']['complete_data'] = df_pre
        st.dataframe(st.session_state['preprocessing_state']['complete_data'])
    
        if 'complete_data' in st.session_state['preprocessing_state']:
        # '장치번호'와 '설비유형_장치List' 가져오기
            filesss = st.file_uploader("Please upload priority opening equipment file", type=['csv', 'xls', 'xlsx'])

        # 업로드된 파일이 있는지 확인
            if filesss is not None:
                # 새로 업로드된 파일 저장
                st.session_state['preprocessing_state']['poe_data'] = filesss

            # 새로 업로드한 장치List 파일을 df로 로드
                if 'poe_data' in st.session_state['preprocessing_state']:
                    # st.write(f"Current File_equipment list: {st.session_state['preprocessing_state']['current_equlist'].name}")
                    poelist = pd.read_excel(st.session_state['preprocessing_state']['poe_data'],  engine='openpyxl')
                    st.session_state['preprocessing_state']['poe_data'] = poelist

                    # '장치번호'를 기준으로 'df_'와 '장치List' 병합
                    poemerged_data = pd.merge(st.session_state['preprocessing_state']['complete_data'], st.session_state['preprocessing_state']['poe_data'], how='left', on='장치번호')

                    # 'poelist의 우선개방' 값을 'completedata의 우선개방여부'에 덮어씌우기
                    st.session_state['preprocessing_state']['complete_data']['우선개방여부'] = poemerged_data['우선개방']
                    st.session_state['preprocessing_state']['complete_data']['우선개방여부'].fillna('해당없음', inplace=True)
                    st.dataframe(st.session_state['preprocessing_state']['complete_data'])
                    st.session_state['eda_state']['current_data'] = st.session_state['preprocessing_state']['complete_data']
        
def preprocessing_page():
    st.title('Preprocessing')
    
    # eda page tab 설정
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2 = st.tabs(['File Merging', 'File Preprocessing'])
    
    with t1:
        file_merging()
    
    with t2:
        file_preprocessing()
        
@st.cache_data
def load_file(file):
    
    # 확장자 분리
    ext = file.name.split('.')[-1]
    
    # 확장자 별 로드 함수 구분
    if ext == 'csv':
        return pd.read_csv(file)
    elif 'xls' in ext:
        return pd.read_excel(file, engine='openpyxl')
        
# get_info 함수
@st.cache_data
def get_info(col, df):
    if col == '장치번호':
        return
    # 독립 변수 1개의 정보와 분포 figure 생성 함수
    plt.figure(figsize=(1.5,1))
    
    # 수치형 변수(int64, float64)는 histogram : sns.histplot() 이용
    if df[col].dtype in ['int64','float64']:
        ax = sns.histplot(x=df[col], bins=30)
        plt.grid(False)
        print(df[col].dtype)
    # 범주형 변수는 seaborn.barplot 이용
    else:
        s_vc = df[col].value_counts().sort_index()
        ax = sns.barplot(x=s_vc.index, y=s_vc.values)

    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('count')
    sns.despine(bottom = True, left = True)
    fig = ax.get_figure()
    
    # 사전으로 묶어서 반환
    return {'name': col, 'total': df[col].shape[0], 'na': df[col].isna().sum(), 'type': df[col].dtype, 'distribution':fig }

# variables 함수
def variables():
    # 각 변수 별 정보와 분포 figure를 출력하는 함수
    
    if 'current_data' in st.session_state['eda_state'] and len(st.session_state['eda_state']['current_data']) > 0:

        # 저장된 df가 있는 경우에만 동작
        if 'current_data' in st.session_state['eda_state']:
            df = st.session_state['eda_state']['current_data']
            cols = df.columns

            # 열 정보를 처음 저장하는 경우 초기 사전 생성
            if 'column_dict' not in st.session_state['eda_state']:
                st.session_state['eda_state']['column_dict'] = {}

            # 모든 열에 대한 정보 생성 후 저장
            for col in cols:
                st.session_state['eda_state']['column_dict'][col] = get_info(col, df)

            # 각 열의 정보를 하나씩 출력
            for col in st.session_state['eda_state']['column_dict']:
                print(f'=============325번째줄 {df[col].dtype}')
                if col == '장치번호':
                    continue
                with st.expander(col, expanded=True):
                    left, right = st.columns((1, 1))
                    right.pyplot(st.session_state['eda_state']['column_dict'][col]['distribution'], use_container_width=True)
                    left.subheader(f"**:blue[{st.session_state['eda_state']['column_dict'][col]['name']}]**")
                    left.caption(st.session_state['eda_state']['column_dict'][col]['type'])
                    # cl, cr = center.columns(2)
                    # cl.markdown('**Missing**')
                    # cr.write(f"{st.session_state['eda_state']['column_dict'][col]['na']}")
                    # cl.markdown('**Missing Rate**')
                    # cr.write(f"{st.session_state['eda_state']['column_dict'][col]['na']/len(df):.2%}")
    else:
        st.warning('You should upload data')
                
# corr 계산 함수
@st.cache_data
def get_corr(options, df):
    # 전달된 열에 대한 pairplot figure 생성
    pairplot = sns.pairplot(df, vars=options)
    return pairplot.fig
            
# correlation tab 출력 함수
def correlation():
    cols = []
    
    # 저장된 df가 있는 경우에만 동작
    if 'current_data' in st.session_state['eda_state']:
        
        df = encodingdata(st.session_state['eda_state']['current_data'])
        st.session_state['eda_state']['current_data'] = df
        cols = df.select_dtypes(['int64', 'float64']).columns
    
    # 상관 관계 시각화를 할 변수 선택 (2개 이상)
    options = st.multiselect(
        'Select the Variables',
        cols,
        [],
        max_selections=len(cols))
    
    # 선택된 변수가 2개 이상인 경우 figure를 생성하여 출력
    if len(options)>=2:
        st.pyplot(get_corr(options, df))
        
def missing_data():
    pass

# EDA 페이지 출력 함수
def eda_page():
    st.title('Data Analysis')
    
    # eda page tab 설정
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2 = st.tabs(['Variables', 'Correlation'])
    
#     with t1:
#         file_uploader()
    
    with t1:
        variables()
    
    with t2:
        correlation()
        
# 독립 변수 선택 및 데이터 분할 함수
def select_split():
    cols = []
    selected_features = []
    selected_label = []
    split_rate = 0
    
    # 저장된 df가 있는 경우에만 실행
    if 'current_data' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data']
        cols = df.columns
    
    # 이미 저장된 선택된 독립 변수가 있으면 그대로 출력
    if 'selected_features' in st.session_state['modeling_state']:
        selected_features = st.session_state['modeling_state']['selected_features']

    # 이미 저장된 선택된 종속 변수가 있으면 그대로 출력
    if 'selected_label' in st.session_state['modeling_state']:
        selected_label = st.session_state['modeling_state']['selected_label']
        
    # 이미 설정된 분할 비율이 있으면 그대로 출력
    if 'split_rate' in st.session_state['modeling_state']:
        split_rate = st.session_state['modeling_state']['split_rate']
    
    # 이미 설정된 랜덤 시드 값이 있으면 그대로 출력
    if 'split_rs' in st.session_state['modeling_state']:
        split_rs = st.session_state['modeling_state']['split_rs']
    
    # 독립 변수 선택
    with st.form('feature_selection'):
        selected_features = st.multiselect(
            'Select the Independent Variables',
            cols,
            selected_features,
            max_selections=len(cols))
        
        submitted = st.form_submit_button('Select')
        if submitted:
            st.session_state['modeling_state']['selected_features'] = selected_features
        st.write(f'Independent Variables: {selected_features}')
    
    # 독립 변수로 선택된 변수 제외
    cols = list(set(cols)-set(selected_features))
    
    # 종속 변수 선택
    with st.form('label_selection'):
        selected_label = st.multiselect(
            'Select the Dependent Variables',
            cols,
            selected_label,
            max_selections=1)
        
        submitted = st.form_submit_button('Select')
        if submitted:
            st.session_state['modeling_state']['selected_label'] = selected_label
        st.write(f'Dependent Variables: {selected_label}')
    # 분할 비율(test_size) 및 랜덤 시드 설정
    with st.form('Split Rate'):
        split_rate = st.slider('Test Rate', 0.1, 0.9, 0.25, 0.01)
        split_rs = st.slider('Random State', 0, 100000, 0, 1)
        
        submitted = st.form_submit_button('Confirm')
        if submitted:
            st.session_state['modeling_state']['split_rate'] = split_rate
            st.session_state['modeling_state']['split_rs'] = split_rs
        st.write(f'Train/Test Ratio → Train: {(1-split_rate):.1%}, Test: {split_rate:.1%}')
        st.write(f'Random Seed: {split_rs}')

# 하이퍼 파라미터 설정
def set_hyperparamters(model_name):
    param_list = {
        'Random Forest Classifier':{
            'n_estimators':[1, 3000, 100, 1], 
            'max_depth':[1, 6, 5, 1],
            'min_samples_leaf':[1, 100, 1, 1],
            'min_samples_split':[2, 100, 2, 1],
            'random_state':[0, 100000, 0, 1]},
        'Gradient Boosting Classifier' :{
            'n_estimators':[1, 2000, 100, 1],
            'max_depth':[1, 6, 5, 1],
            'min_samples_leaf':[1, 100, 1, 1],
            'min_samples_split':[2, 100, 2, 1],
            'subsample':[0.0, 1.0, 1.0, 0.01],
            'learning_rate':[0.0, 1.0, 0.1, 0.01],
            'random_state':[0, 100000, 0, 1]},
        'Extreme Gradient Boosting Classifier':{
            'n_estimators':[1, 2000, 100, 1],
            'max_depth':[1, 6, 5, 1],
            'min_child_weight':[1, 10, 1, 1],
            'subsample':[0.0, 1.0, 1.0, 0.01],
            'colsample_bytree':[0.0, 1.0, 1.0, 0.01],
            'learning_rate':[0.0, 1.0, 0.1, 0.01],
            'random_state':[0, 100000, 0, 1]}
    }
    
    ret = {}
    with st.form('hyperparameters'):
        for key, item in param_list[model_name].items():
            #             # 만약 key가 hyperparameters에 존재한다면 사용
            # if key in st.session_state['modeling_state']['hyperparamters']:
            ret[key] = st.slider(key, *item)
        
        submitted = st.form_submit_button('Run')
        
        if submitted:
            st.write(ret)
            return ret
        
# split data
def split_data():
    df = st.session_state['eda_state']['current_data']
    X = df.loc[:, st.session_state['modeling_state']['selected_features']].values
    Y = df.loc[:, st.session_state['modeling_state']['selected_label']].values.reshape(-1)
    
    test_size = st.session_state['modeling_state']['split_rate']
    seed = st.session_state['modeling_state']['split_rs']
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=test_size)
    
    return x_train, x_test, y_train, y_test
       
# train_model
def train_model(selected_model, model_name):
    with st.spinner('Data Splitting...'):
        x_train, x_test, y_train ,y_test = split_data()
        time.sleep(1)
    st.success('Complete Data Splitting')
    time.sleep(1)

    with st.spinner('Learning...'): 
        model = selected_model(**st.session_state['modeling_state']['hyperparamters'])
        model.fit(x_train, y_train)
    st.success('Complete Learning')

    with st.spinner('Predicting...'):
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
    st.success('Complete Predicting')
    
    file_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 모델 파일 저장
    with open(f'./models/model_{model_name.replace(" ", "_")}_{file_name}.dat', 'wb') as f:
        pickle.dump(model, f)
    
    # 학습에 사용된 독립 변수 목록 저장 (순서)
    with open(f'./models/meta_{model_name.replace(" ", "_")}_{file_name}.dat', 'wb') as f:
        pickle.dump(st.session_state['modeling_state']['selected_features'], f)
        
    return model, y_train, train_pred, y_test, test_pred
        
# modeling 함수
def modeling():
    # 모델링 tab 출력 함수
    model_list = ['Select the Model', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'Extreme Gradient Boosting Classifier']
    model_dict = {'Random Forest Classifier': RandomForestClassifier,
                  'Gradient Boosting Classifier':GradientBoostingClassifier, 
                  'Extreme Gradient Boosting Classifier':XGBClassifier }
    selected_model = ''
    
    if 'selected_model' in st.session_state['modeling_state']:
        selected_model = st.session_state['modeling_state']['selected_model']
    if 'hyperparamters' in st.session_state['modeling_state']:
        hps = st.session_state['modeling_state']['hyperparamters']

    # selected_model = st.multiselect(
    #     'Select the model for learning',
    #     model_list,
    #     [],
    #     max_selections=len(model_list))        

    selected_model = st.selectbox(
        'Select the model for learning.',
        model_list, 
        index=0)

    if selected_model in model_list[1:]:
        st.session_state['modeling_state']['selected_model'] = selected_model
        hps = set_hyperparamters(selected_model)
        st.session_state['modeling_state']['hyperparamters'] = hps
        
        if hps != None:
            model, y_train, train_pred, y_test, test_pred = train_model(model_dict[selected_model], selected_model)
            st.session_state['modeling_state']['model'] = model
            st.session_state['modeling_state']['y_train'] = y_train
            st.session_state['modeling_state']['y_test'] = y_test
            st.session_state['modeling_state']['train_pred'] = train_pred
            st.session_state['modeling_state']['test_pred'] = test_pred
            st.success('Finish Learning')
            
# Confusion Matrix 함수
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predictive")
    plt.ylabel("Actual")
    st.pyplot()

# 결과 tab 함수
def results():
    with st.expander('Metrics', expanded=True):
        if 'y_train' in st.session_state['modeling_state']:
            st.divider()

            d1, d2 = st.columns(2)
            d1.subheader('**:green[Train Results]**')
            left = d1.columns(2)
            ac = accuracy_score(st.session_state['modeling_state']['y_train'], st.session_state['modeling_state']['train_pred'])
            d1.write(f'**:blue[Accuracy]**  :  **{ac: 10.5f}**')

            f1 = f1_score(st.session_state['modeling_state']['y_train'], st.session_state['modeling_state']['train_pred'])
            d1.write(f'**:blue[f1]**  :  **{f1: 10.5f}**')

            rc = recall_score(st.session_state['modeling_state']['y_train'], st.session_state['modeling_state']['train_pred'])
            d1.write(f'**:blue[Recall]**  :  **{rc:10.5f}**')

            pc = precision_score(st.session_state['modeling_state']['y_train'], st.session_state['modeling_state']['train_pred'])
            d1.write(f'**:blue[Precision]**  :  **{pc:10.5f}**')

        if 'y_test' in st.session_state['modeling_state']:
            st.divider()
            d2.subheader('**:green[Test Results]**')
            right = d2.columns(2)

            ac = accuracy_score(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'])
            d2.write(f'**:blue[Accuracy]**  :  **{ac:10.5f}**')

            f1 = f1_score(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'])
            d2.write(f'**:blue[f1]**  :  **{f1:10.5f}**')

            rc = recall_score(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'])
            d2.write(f'**:blue[Recall]**  :  **{rc:10.5f}**')

            pc = precision_score(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'])
            d2.write(f'**:blue[Precision]**  :  **{pc:10.5f}**')

            
    st.divider()
    with st.expander('Classification Report', expanded=False):
        if 'y_train' in st.session_state['modeling_state']:
            st.write('**:blue[0 : 우선개방장치, 1 : 일반장치]**')
            st.table(pd.DataFrame(classification_report(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'], output_dict=True)))
               
    st.divider()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    with st.expander('Result Analysis', expanded=False):
        if 'y_train' in st.session_state['modeling_state']:
            # c1, c2 = st.columns(2)
            st.title('Train')
            plot_confusion_matrix(st.session_state['modeling_state']['y_train'],
                                  st.session_state['modeling_state']['train_pred'],
                                  ['우선개방','非우선개방'])
            st.divider()
            st.title('Test')
            plot_confusion_matrix(st.session_state['modeling_state']['y_test'],
                                  st.session_state['modeling_state']['test_pred'],
                                  ['우선개방','非우선개방'])
     
    st.divider()
    
    with st.expander('Feature Importances', expanded=False):
        if 'model'  in st.session_state['modeling_state']:
            plt.figure()
            plot = sns.barplot(x=st.session_state['modeling_state']['selected_features'],
                               y=st.session_state['modeling_state']['model'].feature_importances_)
            plt.title('Feature Importances')
            plt.xticks(rotation=90)
            fig = plot.get_figure()
            st.pyplot(fig, use_container_width=True)
            
# Modeling 페이지 출력 함수
def modeling_page():
    st.title('Machine Learning')
    
    # tabs를 추가하세요.
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2, t3 = st.tabs(['Data Selection and Split', 'Modeling', 'Results'])

    # file upload tab 구현
    with t1:
        select_split()
    
    with t2:
        modeling()
    
    with t3:
        results()
        
def testfile_merging():
    # 파일 업로더 위젯 추가 (1번쨰)
    material = st.file_uploader("Please upload PSM files", type=['csv', 'xls', 'xlsx'], accept_multiple_files=True)

    if material is not None:
        # 새 파일이 업로드되면 기존 상태 초기화
        st.session_state['serving_state'] = {}
        
        # 새로 업로드된 파일 저장
        st.session_state['serving_state']['current_file'] = material
    else:
        st.warning('Please upload PSM file')
    
    # 새로 업로드한 파일을 df로 로드
    if 'current_data' in st.session_state['serving_state'] and len(st.session_state['serving_state']['current_data']) > 0:
        if st.session_state['serving_state']['current_data']:
            # 첫 번째 파일을 선택 (리스트에서 파일을 선택하는 방식으로 변경)
            selected_material = st.session_state['serving_state']['current_data'][0]

            # 파일 이름 출력
            if hasattr(selected_material, 'name'):
                st.write(f"Current File: {selected_material.name}")
            else:
                st.write("Current File: Does not have 'name' attribute")

            # 선택된 파일을 df로 로드
            st.session_state['serving_state']['current_data'] = load_file(selected_material)
            print(f"708===================={st.session_state['serving_state']['current_data']}")

    # 새로 로드한 df 저장
    if 'current_data' in st.session_state['serving_state']:
        st.dataframe(st.session_state['serving_state']['current_data'])
        
    if st.session_state['serving_state']['current_data'] is not None and len(st.session_state['serving_state']['current_data']) > 0:
        # 업로드된 파일들을 합쳐서 DataFrame으로 가져옴

        combined_testdata = combinedata(st.session_state['serving_state']['current_data'])
        print(f'716--------------------{combined_testdata}')
        # # 결과를 표시
        st.dataframe(combined_data)

        # splitdata 함수 호출
        df_testsplit = splitdata(combined_testdata)
        st.session_state['serving_state']['current_file'] = df_testsplit
        st.session_state['serving_state']['df_split'] = df_testsplit
    # 결과를 표시
        st.dataframe(df_testsplit)
        
# files 변수를 초기화
    materials = []

    if 'df_split' in st.session_state['serving_state'] and len(st.session_state['serving_state']['df_split']) > 0:
        materials = st.file_uploader("Please upload EPS files", type=['csv', 'xls', 'xlsx'], accept_multiple_files=True, key="file_uploader")
    else:
        st.warning('Please upload EPS file')

        # 업로드된 파일이 있는지 확인
        if materials is not None:
            # 새로 업로드된 파일 저장
            st.session_state['serving_state']['current_file'] = materials

            # 업로드된 각 파일을 읽어 DataFrame으로 변환
            dfs_testuploaded = [pd.read_excel(file, header=None) for material in materials]
            if len(dfs_testuploaded) > 0:
                df_testeps = dfs_testuploaded[0]

                st.session_state['serving_state']['current_data'] = df_testeps

                # 새로 로드한 df 저장
                if 'current_data' in st.session_state['serving_state']:

                    df_testeps = df_testeps.rename(columns={0: '장치번호', 1: '설비등급', 2:'설비유형',
                                                              6:'부식율', 7:'잔여수명'})
                    df_testeps[15] = np.nan
                    df_testeps[16] = np.nan
                    df_testeps['우선개방여부'] = '해당없음'   
                    df_testeps.columns = df_testeps.columns.astype(str)
                    st.session_state['serving_state']['eps_data'] = df_testeps

                    # df_a와 df_uploaded를 merge
                    df_testmerged = mergedata(st.session_state['serving_state']['eps_data'], st.session_state['serving_state']['df_split'])
                    st.session_state['serving_state']['eps_data'] = df_testmerged
                    st.dataframe(st.session_state['serving_state']['eps_data'])
                    

    if 'eps_data' in st.session_state['serving_state'] and len(st.session_state['serving_state']['eps_data']) > 0:
        materialss = st.file_uploader("Please upload EQL file", type=['csv', 'xls', 'xlsx'])

        # 업로드된 파일이 있는지 확인
        if materialss is not None:
            # 새로 업로드된 파일 저장
            st.session_state['serving_state']['current_equlist'] = materialss
            
        # 새로 업로드한 장치List 파일을 df로 로드
        if 'current_equlist' in st.session_state['serving_state']:
            # st.write(f"Current File_equipment list: {st.session_state['preprocessing_state']['current_equlist'].name}")
            testequlist = pd.read_excel(st.session_state['serving_state']['current_equlist'], skiprows=1, engine='openpyxl')
            st.session_state['serving_state']['equ_data'] = testequlist
            df_testfinal = insertdata(st.session_state['serving_state']['equ_data'])
            st.session_state['serving_state']['final_data'] = df_testfinal
            # st.dataframe(st.session_state['serving_state']['final_data'])
                    
        if 'final_data' in st.session_state['serving_state'] and len(st.session_state['serving_state']['final_data']) > 0:     
            df_testpre = preprocessdata(st.session_state['serving_state']['final_data'])
            st.session_state['serving_state']['complete_data'] = df_testpre
            st.session_state['serving_state']['current_file'] = st.session_state['serving_state']['complete_data']
            st.dataframe(st.session_state['serving_state']['current_file'])
    
        


#e 데잍 로드하뭇
def loadtestfile(file):
        # 파일이 비어 있는지 확인
    if file is None or file.size == 0:
        st.warning("파일이 비어있습니다.")
        return pd.DataFrame()  # 빈 데이터프레임 반환 또는 예외 처리
    # try:
        # UTF-8로 시도하고 실패하면 다른 인코딩 시도
    print(f'609-----------------------')
    df = pd.read_csv(file, encoding='utf-8')
    print(f'611-----------------------')

#     except UnicodeDecodeError:
#         # 다른 인코딩 시도 (EUC-KR, CP949 등)
#         print(f'615 예외발생..-----------------------')

#         df = pd.read_csv(file, encoding='euc-kr')
    print(f'613 {df}')
    return df

# 추론 함수
def inference():
    model = st.session_state['serving_state']['model']
    model_name = st.session_state['serving_state']['model_name']
    model_name = model_name.removeprefix('model_').replace('_', ' ')
    
    if 'meta' in st.session_state['serving_state']:
        placeholder = ', '.join(st.session_state['serving_state']['meta'])
    else:
        placeholder = ''
    
    with st.expander('Data Upload', expanded=True):
        st.caption(model_name)

        # 새로 업로드한 파일을 df로 로드
        if file is not None:
            st.session_state['serving_state']['current_file'] = file
            # st.write(f"Current File: {st.session_state['serving_state']['current_file'].name}")
            # st.session_state['serving_state']['current_data'] = loadtestfile(st.session_state['serving_state']['current_file'])
            # st.dataframe(st.session_state['serving_state']['current_data'])
        else:
            st.warning("You should upload file")
            # 'current_data'가 없는 경우 여기서 반환하거나 추가 실행을 중단하는 것이 좋습니다.
            # 예를 들어 'return' 또는 'sys.exit()'를 사용할 수 있습니다.
            return
    with st.expander('Summary', expanded=False):
        if 'current_file' in st.session_state['serving_state']:
            dfa = st.session_state['serving_state']['current_file']
        # a = dfa.loc[:, st.session_state['serving_state']['selected_features']].values
        # b = dfa.loc[:, st.session_state['serving_state']['selected_label']].values.reshape(-1)
            pred = model.predict(dfa.iloc[:,1:9])
            print(pred)
            result_df = pd.concat([dfa, pd.Series(pred, name='우선개방여부')], axis=1)
            result_df['우선개방여부'] = result_df['우선개방여부'].replace({'우선개방': '해당', '해당없음': '미해당'})
            st.session_state['serving_state']['current_data'] = result_df
        
            # 결과 DataFrame 출력
            st.dataframe(result_df)
        else:
            st.warning("You should upload file")
            
    with st.expander('Result', expanded=True):
        if 'current_data' in st.session_state['serving_state']:
            # '우선개방여부' 열이 '우선개방대상'인 행만 추출하여 별도로 정렬
            prioritized_rows = result_df[result_df['우선개방여부'] == '해당']

            # 정렬된 DataFrame 출력
            st.subheader('**:red[Priority Opening Fixed Equipment]**')
            st.dataframe(prioritized_rows.loc[:,['장치번호','우선개방여부']])
            st.write(f"**:blue[총 {len(result_df)}개의 장치 중 {len(prioritized_rows)}개의 장치가 우선개방대상으로 분류되었습니다.]**")

# Serving 함수
def serving():    
    with st.form('select pre-trained model'):
        # 모델 파일 목록 불러오기
        model_paths = glob.glob('./models/model_*')
        model_paths.sort(reverse=True)
        model_list = [s.removeprefix('./models/').removesuffix('.dat') for s in model_paths]
        model_dict = {k:v for k, v in zip(model_list, model_paths)}
        model_list = ['Select Model'] + model_list

        # 추론에 사용할 모델 선택

        selected_inference_model = st.selectbox('Select the Model for Inference', model_list, index=0)
        checked = st.checkbox('Independent Variables Information')

        submitted = st.form_submit_button('Confirm')

        if submitted:
            st.session_state['serving_state'] = {}
            with open(model_dict[selected_inference_model], 'rb') as f_model:
                inference_model = pickle.load(f_model)
                st.session_state['serving_state']['model'] = inference_model
                st.session_state['serving_state']['model_name'] = selected_inference_model
                if checked:
                    with open(model_dict[selected_inference_model].replace('model_', 'meta_'), 'rb') as f_meta:
                        metadata = pickle.load(f_meta)
                        st.session_state['serving_state']['meta'] = metadata
                placeholder = st.empty()
                placeholder.success('Success')
                time.sleep(2)
                placeholder.empty()

    if 'model' in st.session_state['serving_state']:
        inference()
            
# Serving 페이지 출력 함수
def serving_page():
    st.title('Model Serving')
    
    # eda page tab 설정
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2 = st.tabs(['Test File Preprocessing', 'Model Serving'])
    
    with t1:
        testfile_merging()
    
    with t2:
        serving()



        
# session_state에 사전 sidebar_state, eda_state, modeling_state, serving_state를 추가하세요.
if 'sidebar_state' not in st.session_state:
    st.session_state['sidebar_state'] = {}
    st.session_state['sidebar_state']['current_page'] = front_page
if 'preprocessing_state' not in st.session_state:
    st.session_state['preprocessing_state'] = {}
if 'eda_state' not in st.session_state:
    st.session_state['eda_state'] = {}
if 'modeling_state' not in st.session_state:
    st.session_state['modeling_state'] = {}
if 'serving_state' not in st.session_state:
    st.session_state['serving_state'] = {}
    
# sidebar 추가 preprocessing_page
with st.sidebar:
    st.subheader('Contents')
    b1 = st.button('Main Page', use_container_width=True)
    b2 = st.button('Preprocessing Page', use_container_width=True)
    b3 = st.button('EDA Page', use_container_width=True)
    b4 = st.button('Modeling Page', use_container_width=True)
    b5 = st.button('Serving Page', use_container_width=True)
    
if b1:
    st.session_state['sidebar_state']['current_page'] = front_page
#     st.session_state['sidebar_state']['current_page']()
    front_page()
elif b2:
    st.session_state['sidebar_state']['current_page'] = preprocessing_page
#     st.session_state['sidebar_state']['current_page']()
    preprocessing_page()    
elif b3:
    st.session_state['sidebar_state']['current_page'] = eda_page
#     st.session_state['sidebar_state']['current_page']()
    eda_page()
elif b4:
    st.session_state['sidebar_state']['current_page'] = modeling_page
#     st.session_state['sidebar_state']['current_page']()
    modeling_page()
elif b5:
    st.session_state['sidebar_state']['current_page'] = serving_page
#     st.session_state['sidebar_state']['current_page']()
    serving_page()
else:
    st.session_state['sidebar_state']['current_page']()
