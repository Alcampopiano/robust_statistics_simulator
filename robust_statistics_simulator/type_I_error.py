import streamlit as st
from funcs.simulation import make_pdf, make_population_chart, dists, make_sampling_distribution_of_t_chart, \
    t_sampling_distribution_loop, make_type_I_error_chart, type_I_error_button_callback

#######################################

st.title('Population Characteristics')
st.markdown('<br>', unsafe_allow_html=True)
shape_param=st.selectbox('Choose population shape', dists)
st.markdown('<br>', unsafe_allow_html=True)

if shape_param in ['contaminated normal', 'contaminated chi-squared']:
    scale_param = st.slider('Choose contamination', .01, 1.0, .1, step=.01)
else:
    scale_param = st.slider('Choose standard deviation', 1.0, 5.0, 1.0, step=.01)

df=make_pdf(scale_param, shape_param)
c=make_population_chart(df)
st.write(c)

######################################

st.title('Sampling distribution of t')
samp_param = st.slider('Choose sample size', 1, 100, 20, step=1)
sample=t_sampling_distribution_loop(scale_param, shape_param, samp_param)
c=make_sampling_distribution_of_t_chart(sample, samp_param)
st.write(c)

st.title('Type I Error Estimates')
st.markdown('<br>', unsafe_allow_html=True)
g_param = st.slider('Skewness', 0.0, 1.0, 0.0, step=0.1)
h_param = st.slider('Heaviness', 0.0, .9, 0.0, step=0.1)

if st.button('Run simulations'):
    results=type_I_error_button_callback(g_param,h_param)
    c3=make_type_I_error_chart(results)
    st.write(c3)









