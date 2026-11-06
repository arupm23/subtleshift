const arms = {
  base: { index:'00', kicker:'REFERENCE CONDITION', title:'Untouched base model', description:'Measures the model’s pre-fine-tuning behavior under the exact evaluation and decoding protocol used for every adapted arm.', adapter:'None', data:'None', role:'Reference', label:'BASELINE' },
  benign: { index:'01', kicker:'MATCHED-DOMAIN CONTROL', title:'Benign helpfulness', description:'Controls for supervised fine-tuning and domain exposure using safe, helpful examples with the same training budget.', adapter:'LoRA', data:'Benign domain', role:'SFT control', label:'BENIGN' },
  neutral: { index:'02', kicker:'FORMAT CONTROL', title:'Neutral style transfer', description:'Tests whether generic formatting and style learning alone changes evaluation behavior outside the training distribution.', adapter:'LoRA', data:'Neutral style', role:'Format control', label:'NEUTRAL' },
  treatment: { index:'03', kicker:'PRE-REGISTERED TEST', title:'Reviewed treatment set', description:'Tests the target training intervention using an approved, versioned dataset under the same optimizer, token budget, and seed schedule.', adapter:'LoRA', data:'Reviewed research set', role:'Primary test', label:'TREATMENT' }
};

const tabs = [...document.querySelectorAll('[role="tab"]')];
const panel = document.querySelector('#arm-panel');
function selectArm(key){
  const arm = arms[key];
  tabs.forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.arm === key)));
  panel.setAttribute('aria-labelledby', `tab-${key}`);
  panel.animate([{opacity:.35,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:260,easing:'ease-out'});
  document.querySelector('#arm-index').textContent=arm.index;
  document.querySelector('#arm-kicker').textContent=arm.kicker;
  document.querySelector('#arm-title').textContent=arm.title;
  document.querySelector('#arm-description').textContent=arm.description;
  document.querySelector('#arm-adapter').textContent=arm.adapter;
  document.querySelector('#arm-data').textContent=arm.data;
  document.querySelector('#arm-role').textContent=arm.role;
  document.querySelector('#arm-visual-label').textContent=arm.label;
}
tabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>selectArm(tab.dataset.arm));
  tab.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight'].includes(event.key))return;
    event.preventDefault();
    const next=(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    tabs[next].focus(); selectArm(tabs[next].dataset.arm);
  });
});

const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{
  if(entry.isIntersecting){entry.target.classList.add('visible');observer.unobserve(entry.target)}
}),{threshold:.12});
document.querySelectorAll('.reveal').forEach(el=>observer.observe(el));
