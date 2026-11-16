"""Synthetic SAP S/4HANA UAT capstone demo. No client data."""
import io
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODEL = ROOT / 'models'
st.set_page_config(page_title='Synthetic SAP UAT Intelligence', page_icon='🧪', layout='wide')

@st.cache_data
def read_csv(name, folder='data'):
    return pd.read_csv((DATA if folder == 'data' else MODEL) / name).fillna('')

@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL / 'defect_pipeline.joblib')

def api_key():
    try:
        return st.secrets.get('GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')
    except Exception:
        return os.environ.get('GEMINI_API_KEY', '')

def gemini(prompt, as_json=False):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key())
    model_name = st.session_state.get('gemini_model', 'gemini-3.8-flash')
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=3500,
            **({'response_mime_type': 'application/json'} if as_json else {}),
        ),
    )
    if not response.text:
        raise ValueError('Gemini returned no text')
    return response.text

def init_review():
    if 'review_df' not in st.session_state:
        frame = read_csv('generated_uat_cases.csv').copy()
        frame['review_status'] = 'Pending human review'
        frame['reviewer_comments'] = frame.get('reviewer_comments', '')
        st.session_state.review_df = frame

st.title('🧪 AI-Powered SAP UAT Intelligence')
st.caption('Academy capstone demonstration · 100% synthetic data · No operational SAP or Defence client information')
st.warning('Synthetic research demonstration only. ML confidence is not calibrated, Gemini output requires human review, and regression exposure is retrospective.', icon='⚠️')

page = st.sidebar.radio('Navigate', ['Dashboard', 'GenAI test generation', 'Review and approve', 'Defect classification', 'Regression optimisation', 'Contextual assistant', 'Evidence and limitations'])
st.sidebar.caption('All supplied records are fictional. Do not enter real client requirements, defects or credentials.')
st.sidebar.text_input('Gemini model ID', value='gemini-3.8-flash', key='gemini_model')

if page == 'Dashboard':
    req = read_csv('requirements.csv')
    defects = read_csv('defects.csv')
    tests = read_csv('test_cases.csv')
    comparison = read_csv('regression_comparison.csv')
    risk = comparison.loc[comparison.method == 'risk_based'].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Synthetic requirements',len(req))
    c2.metric('Synthetic defects',len(defects))
    c3.metric('Reference test cases',len(tests))
    c4.metric('Regression effort saved',f"{float(risk.effort_reduction_pct):.2f}%")
    st.subheader('Requirements by business process')
    st.bar_chart(req.groupby('process_name').size().rename('Requirements'))
    st.subheader('ML evaluation (held-out synthetic examples)')
    st.dataframe(read_csv('holdout_model_comparison.csv','models'),hide_index=True,use_container_width=True)
    st.caption('The training-only cross-validation selected Logistic Regression; holdout metrics are not production performance.')
    st.subheader('Regression strategy comparison')
    st.dataframe(comparison,hide_index=True,use_container_width=True)

elif page == 'GenAI test generation':
    req = read_csv('requirements.csv')
    st.subheader('Generate structured UAT candidates from a fictional requirement')
    choice = st.selectbox('Synthetic requirement',req.requirement_id.tolist())
    record = req.loc[req.requirement_id == choice].iloc[0]
    st.write('**Requirement:**',record.requirement_text)
    st.write('**Acceptance criteria:**',record.acceptance_criteria)
    st.caption('The saved V2-03 cases below were generated through a prior live Gemini run. New generation requires your configured API key and may incur charges.')
    prior = read_csv('generated_uat_cases.csv')
    st.dataframe(prior.loc[prior.requirement_id == choice],hide_index=True,use_container_width=True)
    if st.button('Generate 2 NEW Gemini candidates',disabled=not bool(api_key())):
        prompt = f'''You are creating UAT candidates ONLY for a fictional SAP S/4HANA-like process. Never invent transaction codes, Fiori apps, configuration or guaranteed workflow behaviour. State assumptions as review items. Generate exactly two distinct scenarios, one positive and one negative. Return ONLY JSON as an array of two objects, each with keys test_scenario, scenario_type, preconditions, test_steps, expected_result, configuration_assumptions. Every expected_result must describe one unambiguous observable outcome. Requirement ID: {record.requirement_id}. Process: {record.process_name}. Requirement: {record.requirement_text}. Acceptance criteria: {record.acceptance_criteria}. All content must remain synthetic.'''
        try:
            with st.spinner('Calling Gemini...'):
                payload=json.loads(gemini(prompt,as_json=True))
            if isinstance(payload,dict):
                payload=payload.get('test_cases',payload.get('cases',[]))
            if not isinstance(payload,list) or len(payload)!=2:
                raise ValueError('Expected exactly two JSON cases')
            needed={'test_scenario','scenario_type','preconditions','test_steps','expected_result','configuration_assumptions'}
            if any(not isinstance(c,dict) or not needed.issubset(c) for c in payload):
                raise ValueError('Required test fields missing')
            generated=pd.DataFrame(payload)
            generated.insert(0,'requirement_id',choice)
            generated.insert(0,'test_case_id',[f'LIVE-{choice}-01',f'LIVE-{choice}-02'])
            generated['review_status']='Pending human review'
            generated['reviewer_comments']=''
            st.session_state.new_cases=generated
            st.success('Two candidates generated. Review before approval.')
        except Exception as exc:
            st.error(f'Generation failed: {type(exc).__name__}: {exc}')
    if 'new_cases' in st.session_state:
        st.dataframe(st.session_state.new_cases,hide_index=True,use_container_width=True)
        st.download_button('Download newly generated candidates',st.session_state.new_cases.to_csv(index=False),'new_gemini_candidates.csv','text/csv')
        if st.button('Add new candidates to review queue'):
            init_review()
            old=st.session_state.review_df
            fresh=st.session_state.new_cases.copy()
            for col in old.columns:
                if col not in fresh: fresh[col]=''
            st.session_state.review_df=pd.concat([old,fresh[old.columns]],ignore_index=True)
            st.success('Added to this session’s review queue. Export approved work before refreshing.')
    if not api_key():
        st.info('To enable live generation, add GEMINI_API_KEY in Streamlit app secrets. Existing V2-03 evidence remains viewable without a key.')

elif page == 'Review and approve':
    st.subheader('Human-in-the-loop test case review')
    init_review()
    st.caption('Changes are held in this browser session; download the CSV to retain decisions. No permanent database is connected.')
    editable=['test_scenario','preconditions','test_steps','expected_result','acceptance_criteria','review_status','reviewer_comments']
    disabled=[c for c in st.session_state.review_df.columns if c not in editable]
    editor=st.data_editor(st.session_state.review_df,hide_index=True,use_container_width=True,num_rows='fixed',disabled=disabled,
                          column_config={'review_status':st.column_config.SelectboxColumn('Review status',options=['Pending human review','Approved','Rejected','Needs revision'],required=True)},key='review_editor')
    if st.button('Apply review decisions'):
        invalid=editor.loc[(editor.review_status=='Approved') & ((editor.test_steps.astype(str).str.strip()=='') | (editor.expected_result.astype(str).str.strip()==''))]
        if len(invalid):st.error('Cannot approve cases with empty steps or expected results.')
        else:
            st.session_state.review_df=editor.copy()
            st.success('Decisions applied to this session. Download CSV to preserve them.')
    st.download_button('Download reviewed UAT cases',editor.to_csv(index=False),'reviewed_uat_cases.csv','text/csv')
    st.write('Review statuses (currently shown in editor):')
    st.dataframe(editor.review_status.value_counts().rename_axis('Status').reset_index(name='Count'),hide_index=True)

elif page == 'Defect classification':
    st.subheader('Supervised defect triage')
    st.caption('Logistic Regression selected by train-only CV; trained on 816 synthetic defects, evaluated on 384 held-out synthetic defects. Do not treat probability as calibrated certainty.')
    description=st.text_area('Describe a fictional defect','In a fictional approval workflow, the configured threshold was not applied during submission.')
    if st.button('Classify synthetic defect'):
        if not description.strip():st.error('Enter a description first.')
        else:
            try:
                pipeline=load_model()
                label=pipeline.predict([description])[0]
                st.success(f'Predicted triage category: {label}')
                if hasattr(pipeline,'predict_proba'):
                    probs=pipeline.predict_proba([description])[0]
                    categories=pipeline.classes_
                    out=pd.DataFrame({'Category':categories,'Model probability (uncalibrated)':probs}).sort_values('Model probability (uncalibrated)',ascending=False)
                    st.dataframe(out,hide_index=True,use_container_width=True)
                    st.caption('Probabilities are model outputs, not calibrated operational confidence.')
            except Exception as exc:
                st.error(f'Model could not be loaded or used: {type(exc).__name__}: {exc}')
    st.subheader('Held-out model comparison')
    st.dataframe(read_csv('holdout_model_comparison.csv','models'),hide_index=True,use_container_width=True)

elif page == 'Regression optimisation':
    st.subheader('Explainable, coverage-constrained regression pack')
    comparison=read_csv('regression_comparison.csv')
    st.dataframe(comparison,hide_index=True,use_container_width=True)
    risk=comparison.loc[comparison.method=='risk_based'].iloc[0]
    c1,c2,c3=st.columns(3)
    c1.metric('Selected tests',int(risk.selected_tests))
    c2.metric('Estimated effort saved',f'{float(risk.effort_reduction_pct):.2f}%')
    c3.metric('Historical exposure retained',f'{float(risk.historical_exposure_retained_pct):.2f}%')
    st.bar_chart(comparison.set_index('method')[['selected_minutes']])
    st.caption('All constrained methods were compared at approximately the same execution-time budget. Risk and historical exposure are synthetic, retrospective estimates.')
    st.subheader('Selection rationale')
    selected=read_csv('selected_tests_risk_based.csv')
    proc=st.selectbox('Filter process',['All']+sorted(selected.process_code.astype(str).unique().tolist()))
    filtered=selected if proc=='All' else selected.loc[selected.process_code==proc]
    st.dataframe(filtered[['test_case_id','requirement_id','process_code','risk_score','minutes','selection_reason','risk_explanation']],hide_index=True,use_container_width=True)
    st.download_button('Download selected regression pack',selected.to_csv(index=False),'selected_regression_pack.csv','text/csv')
    st.subheader('Coverage validation')
    st.dataframe(read_csv('validation_checks.csv'),hide_index=True,use_container_width=True)
    st.info('Only V2-01 reference tests are in this precomputed regression pack; pending Gemini candidates are excluded.')

elif page == 'Contextual assistant':
    st.subheader('Synthetic UAT contextual assistant')
    st.caption('The assistant receives only the selected synthetic requirement and your question; it does not connect to SAP.')
    req=read_csv('requirements.csv')
    selected_id=st.selectbox('Context requirement',req.requirement_id.tolist())
    record=req.loc[req.requirement_id==selected_id].iloc[0]
    question=st.text_area('Ask about the fictional requirement or testing approach','What positive and negative test conditions should a reviewer consider?')
    if st.button('Ask Gemini',disabled=not bool(api_key())):
        try:
            with st.spinner('Generating answer...'):
                answer=gemini(f'''Answer only about this fictional SAP-like UAT requirement. Do not invent actual SAP configuration, transactions or client facts. Separate requirement facts from assumptions. Requirement: {record.requirement_text}. Acceptance criteria: {record.acceptance_criteria}. User question: {question}''')
            st.write(answer)
        except Exception as exc:st.error(f'Assistant request failed: {type(exc).__name__}: {exc}')
    if not api_key():st.info('Add GEMINI_API_KEY in Streamlit secrets to enable this live assistant.')

else:
    st.subheader('Reproducible evidence and limitations')
    st.markdown('''**V2-01:** 120 fictional requirements, 1,200 labelled synthetic defects, 480 reference tests.  
**V2-02:** Group-aware holdout split; 816 train / 384 test. Logistic Regression selected by train-only CV, not by holdout performance.  
**V2-03:** Four live Gemini-generated candidates from two synthetic requirements; structural checks passed, human approval remains required.  
**V2-04:** 352/480 tests selected; 30.02% estimated execution effort reduction, 100% requirement/process/mandatory-critical coverage; 82.13% retrospective synthetic historical exposure retained.  
**Security:** No confidential client data, SAP connections, or secrets are committed. API keys belong in Streamlit secrets.  
**Limits:** Synthetic text can leak generation patterns; the classifier is not validated on real defects; risk metrics are not future defect-detection guarantees; Streamlit session edits are not persistent; generated cases require domain review.''')
    st.write('Synthetic sample requirements')
    st.dataframe(read_csv('requirements.csv').head(8),hide_index=True,use_container_width=True)
