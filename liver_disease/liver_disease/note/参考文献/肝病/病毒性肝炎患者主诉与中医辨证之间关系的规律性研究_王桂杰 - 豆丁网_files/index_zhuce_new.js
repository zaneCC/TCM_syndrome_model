//首页v4_1版 注册验证

var index_reg_name_new = false;
var index_reg_email_new = false;
var index_reg_password_new = false;
var index_reg_rand_new = false;

function hasClass(ele,cls) {
  return ele.className.match(new RegExp('(\\s|^)'+cls+'(\\s|$)'));
}
 
function addClass(ele,cls) {
  if (!this.hasClass(ele,cls)) ele.className += " "+cls;
}
 
function removeClass(ele,cls) {
  if (hasClass(ele,cls)) {
          var reg = new RegExp('(\\s|^)'+cls+'(\\s|$)');
    ele.className=ele.className.replace(reg,' ');
  }
}
/*----------选项卡--------*/
function loginSinaSwitch(selfObj,boxObj){                   
	var lis=document.getElementById(selfObj).getElementsByTagName("li");
	for(var i=0,j=lis.length;i<j;i++){
		document.getElementById(boxObj+i).style.display="none";
		
		if(!!hasClass(lis[i],"cur")){
			document.getElementById(boxObj+i).style.display="block";
		}
		lis[i].onmousedown=function(){
			if(this.className=="cur"){
				return false;
			}
			else{
				thiSwitch(this);
			}
		}
	}
	function thiSwitch(obj){         //thiSwitch
		for(var i=0,j=lis.length;i<j;i++){
			if (lis[i]==obj){
				 addClass(lis[i],"cur");
				 document.getElementById(boxObj+i).style.display="block";
			}
			else{
				 removeClass(lis[i],"cur");
				 document.getElementById(boxObj+i).style.display="none";
			}
		}
	}
}
function showlogin(){
	jQuery("#loginwindow").load("/jsp_cn/weibo/login.jsp",function (data){
		dialogBoxShadow();
		//document.getElementById("newlogin").style.display="block";
		if(document.getElementById("newlogin")){
			setObjCenter("newlogin");
			jQuery('input[placeholder]').placeholder();
			jQuery('input[placeholder]').bind("focus",function(){
				jQuery(this).addClass("txtFocus");
			});
			jQuery('input[placeholder]').bind("blur",function(){
				jQuery(this).removeClass("txtFocus");
			});
			jQuery("#username_new").mailAutoComplete();
		}
		document.getElementById("username_new").focus();
	});
}
function closeShowMessage_sina(){
dialogBoxHidden();
document.getElementById("newlogin").style.display="none";
document.getElementById("bangding").style.display="none";
}
function closeShowMessage_first(){
dialogBoxHidden();
document.getElementById("newlogin").style.display="none";
}
function openss_back(){
//jQuery("#open").load("/jsp_cn/weibo/call.jsp?opt=1");
}
/***微信开始**/
var interval;
function openwx(url){
	if(url==null || url==''){
		url=window.location.href;
	}
	jQuery("#wxwindow").load("jsp_cn/weixin/weixinModel.jsp?url="+url,function(data){
		dialogBoxShadow();
		if(document.getElementById("microsoft_xin")){
			setObjCenter("microsoft_xin");
			openwx_back();
		}
	});
}
function closeWeixin(){
	dialogBoxHidden();
	document.getElementById("microsoft_xin").style.display="none";
	clearInterval(interval);
}
//微信扫码登录
function openwx_back(){
	 jQuery.post("/weixin/getQRTicket.do",{},function(data){
			if(data != null && data.length > 0){
				jQuery("#login_img").attr("src", "https://mp.weixin.qq.com/cgi-bin/showqrcode?ticket="+escape(data));
				//监听扫码状态
				interval = setInterval("validateQR('"+data+"')",3000);
				setTimeout("clearInterval('"+interval+"')",120000);
			}
		});
}
// 监听二维码扫描状态
function validateQR(ticket){
	jQuery.post("/weixin/validateQR.do",{"ticket":ticket},function(data){
		if(data != null && data.length > 0){
				clearInterval(interval);
				//处理登录相关
				jQuery("#wxwindow").load("jsp_cn/weixin/weixinCallback.jsp?user_id="+escape(data),function(data){});
		}
	});
}
/***微信结束**/
function openss(url){
	if(url==null || url==''){
		url=window.location.href;
	}
  window.open('/jsp_cn/weibo/callv2.jsp?url='+url,'new','height='+450+',,innerHeight='+450+',width='+550+',innerWidth='+550+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

/************ 验证注册邮箱是否正确并存在 *********************/
function index_checkRegName_new(type){
	document.getElementById("showBlurUserName_new").className = "val dpb";
	var loginName = trim(document.getElementById("regloginname_new").value);
	var desc = '';
	
	
	if(loginName==''){
		desc = "用户名不能为空，请填写！";
	}
	var c = new RegExp();   
	c = /^[A-Za-z0-9_-]+$/;    
	if(!c.test(loginName)){
		desc = "用户名只支持英文 数字的组合，请正确填写!";
	}
	if(loginName.length>20){
		desc = "用户名的长度不能大于20!"
	}  
	if(loginName.toUpperCase().indexOf("DOCIN")>-1 || loginName.toUpperCase().indexOf("VONIBO")>-1 || loginName.toUpperCase().indexOf("BBS")>-1 ){
		desc = "该用户名已经存在，请重新输入！";
	}
	if(desc != ""){
		
			document.getElementById("showBlurUserName_new").src = picture_image_path_v1+"/images_cn/registration/reg_icon_cw.gif";
			document.getElementById("showBlurUserName_new").title = desc;
		document.getElementById("showBlurUserName_new").style.display = "block";
		return;
	}else{
		index_reg_name_new = true;
	}
	
	if(index_reg_name_new){
		//checkLoginDwr.checkLoginName(loginName,
			jQuery("#a5").load('/jsp_cn/jquery/login/reg_check.jsp?flag=name&loginName='+loginName,
			function (data){
				data = trim(data.replace(/\r\n/gim, "")); 
				if(data=="true"){
					
						document.getElementById("showBlurUserName_new").src = picture_image_path_v1+"/images_cn/registration/reg_zq.gif";
						document.getElementById("showBlurUserName_new").title = "";
					document.getElementById("showBlurUserName_new").style.display = "block";
				}else{
				
						document.getElementById("showBlurUserName_new").src = picture_image_path_v1+"/images_cn/registration/reg_icon_cw.gif";
						document.getElementById("showBlurUserName_new").title = "该用户名已经存在，请重新输入";
					document.getElementById("showBlurUserName_new").style.display = "block";
				}
			}
		);
	}
}
function index_checkRegEmail_new(type){
	document.getElementById("showBlurEmail_new").className = "val dpb";
	var loginEmail = document.getElementById("regloginemail_new").value;
	var desc='';
	if(loginEmail.indexOf(" ")>-1){
		desc='邮箱不能包含空格!';
	}else {
		loginEmail=trim(loginEmail);
		if(loginEmail==''){
			desc='请您输入邮箱!';
		}else if(!checkEmail(loginEmail)){
			desc='邮箱格式不正确,请重新输入';
		}else if (new RegExp("[,]","g").test(loginEmail)){
			desc='含有非法字符'
		}else if(loginEmail.length>100){
			desc='邮箱长度应小于100个字符';
		}
	}
	if(desc!=''){
		
			document.getElementById("showBlurEmail_new").src = picture_image_path_v1+"/images_cn/registration/reg_icon_cw.gif";
			document.getElementById("showBlurEmail_new").title = desc;
			document.getElementById("showBlurEmail_new").style.display = "block";
		return;
	}else{
		index_reg_email_new = true;
	}
	if(index_reg_email_new){
		loginEmail = trim(loginEmail);
		
		jQuery("#a5").load('/jsp_cn/jquery/login/reg_check.jsp?flag=email&loginEmail='+loginEmail,
			function (data){
				data = trim(data.replace(/\r\n/gim, "")); 
				if(data=="true"){
						document.getElementById("showBlurEmail_new").src = picture_image_path_v1+"/images_cn/registration/reg_zq.gif";
						document.getElementById("showBlurEmail_new").style.display = "block";
					
				}else{
						document.getElementById("showBlurEmail_new").src = picture_image_path_v1+"/images_cn/registration/reg_icon_cw.gif";
						document.getElementById("showBlurEmail_new").title = "该邮箱已经存在，请重新输入";
						document.getElementById("showBlurEmail_new").style.display = "block";
					
				}
			}
		);
		
		
	}
}
/************ 验证密码是否输入正确 *********************/
function index_checkRegPassword_new(type){
	document.getElementById("showBlurPwd_new").className = "val dpb";
	var v = document.getElementById("regpassword_new").value;
	var desc='';
	if(v.indexOf(" ")>-1){
		desc='密码不能包含空格';
	}else{
		v=trim(v);
		var c = new RegExp();   
		c = /^[A-Za-z0-9_\-`~!@#\$%^&\*\(\)\+=\{\}\[\]\|:;"'<>,\.\?\\\/]+$/;  
		if(v==''){
			desc='请您输入密码';
		}else if(v.length<6){
			desc='密码长度不能小于6';
		}else if(v.length>16){
			desc='密码长度不能大于16';
		}else if(!c.test(v)){
			desc = "支持英文/数字/符号组合，请正确填写!";
		}
	}
	if(desc!=''){
			document.getElementById("showBlurPwd_new").src = picture_image_path_v1+"/images_cn/registration/reg_icon_cw.gif";
			document.getElementById("showBlurPwd_new").title = desc;
			document.getElementById("showBlurPwd_new").style.display = "block";
		return;
	}else{
			document.getElementById("showBlurPwd_new").src = picture_image_path_v1+"/images_cn/registration/reg_zq.gif";
			document.getElementById("showBlurPwd_new").style.display = "block";
		index_reg_password_new=true;
		return;
	}
}
/************ 验证验证码 *********************/
function index_checkCode_new(){
	var code = encodeURIComponent(trim(document.getElementById("yanzhengma_new").value));
	if(code==""){	alert("请输入验证码");	return;	}
	//checkLoginDwr.checkCode(code,index_showCheckCodeResult);
	jQuery("#a5").load('/jsp_cn/jquery/login/reg_check.jsp?flag=code&code='+code,index_showCheckCodeResult);
}
function index_showCheckCodeResult(data){
	data = trim(data.replace(/\r\n/gim,""));
	if(data=="true"){
		index_reg_rand_new = true;
		
	}else{
		alert("验证码输入错误，请重新输入");
		index_reg_rand_new = false;
		refCode_new();
	}
}


function index_submit(){
	if(index_reg_name_new  && index_reg_email_new && index_reg_password_new && index_reg_rand_new ){
		return true;
	}else{
		return false;
	}
}
function refCode_new(){
	// 注释掉的为原来的汉子验证码
/*	var d, s = "";
	var c = "";
	d = new Date();
	s += d.getYear()+c;
	s += (d.getMonth() + 1) + c;
	s += d.getDate() + c;
	s += d.getHours() + c;
	s += d.getMinutes() + c;
	s += d.getSeconds() + c;
	s += d.getMilliseconds();
//	$('regimg_new').src="/servlet/getctime?"+s;
	document.getElementById("regimg_new").src ="/servlet/getctime?"+s;*/
	var regCodeImg = jQuery('#regimg_new');
	if(regCodeImg.length>0){
		regCodeImg.attr('src',"/servlet/getctime?from=addusercode&t=" + new Date().getTime());
	}
}
function setWb_News(){
	var num = "0";
	var obj = document.getElementsByName('change');
	for(i=0;i<obj.length;i++){
		if(obj[i].checked){
			num = "1";
			break;
		}
	}
	if(num == "0"){
		alert("至少选择一项操作");
		return;
	}else{
		submitWb_News();
	}
}
function submitWb_News(){
	var url = "/jsp_cn/weibo/update.jsp?time="+new Date().getTime();
	//var url = "/app/rr/aa?time="+new Date().getTime()+"&fn=setrr";

	if(document.getElementById("change_down").checked){
		url += "&cd=1";
	}
	if(document.getElementById("change_fav").checked){
		url += "&cf=1";
	}
	if(typeof(document.getElementById("change_bbs")) != 'undefined' && document.getElementById("change_bbs") != null && document.getElementById("change_bbs").checked){
		url += "&cb=1";
	}
	if(document.getElementById("change_message").checked){
		url += "&cm=1";
	}
	if(typeof(document.getElementById("change_doudan_fav")) != 'undefined' && document.getElementById("change_doudan_fav") != null && document.getElementById("change_doudan_fav").checked){
		url += "&cdf=1";
	}
	if(typeof(document.getElementById("change_doudan_top")) != 'undefined' && document.getElementById("change_doudan_top") != null && document.getElementById("change_doudan_top").checked){
		url += "&cdt=1";
	}
	jQuery("#setwb").load(url);
}
function setSD_News(){
	var num = "0";
	var obj = document.getElementsByName('change');
	for(i=0;i<obj.length;i++){
		if(obj[i].checked){
			num = "1";
			break;
		}
	}
	if(num == "0"){
		alert("至少选择一项操作");
		return;
	}else{
		submitSD_News();
	}
}
function submitSD_News(){
	var url = "/jsp_cn/shengda/update.jsp?time="+new Date().getTime();
	//var url = "/app/rr/aa?time="+new Date().getTime()+"&fn=setrr";

	if(document.getElementById("change_down").checked){
		url += "&cd=1";
	}
	if(document.getElementById("change_fav").checked){
		url += "&cf=1";
	}
	if(typeof(document.getElementById("change_bbs")) != 'undefined' && document.getElementById("change_bbs") != null && document.getElementById("change_bbs").checked){
		url += "&cb=1";
	}
	if(document.getElementById("change_message").checked){
		url += "&cm=1";
	}
	jQuery("#setwb").load(url);
}
function jiechu(){
	jQuery("#jiechuwb").load("/jsp_cn/weibo/jiechu.jsp?time="+new Date().getTime());
	
}

function checkInfo(){
	var u=document.getElementById("username_isdoc").value;
	var p=document.getElementById("password_isdoc").value;
	if(trim(u)=="" || trim(p)==""){
		alert("用户名/密码不能为空");
		return false;
	}
	return true;
	
}

function check_login(){
	var u=document.loginnew.username.value;
	var p=document.loginnew.password.value;
	if(trim(u)=="" || trim(p)==""){
		alert("用户名/密码不能为空");
		return false;
	}
	var reg = /[\u4E00-\u9FA5\uF900-\uFA2D]/;    
	if(reg.test(u)){
		alert("用户名只支持英文 数字的组合，请正确填写!");
		return false;
	}
	return true;
	
	
}
function showsub(){
	
	jQuery("#loginwindow").load("/jsp_cn/weibo/sub_login.jsp?time="+new Date().getTime());
}

function loginout_sd(){
	jQuery("#setwb").load("/jsp_cn/shengda/loginout.jsp?time="+new Date().getTime());
	
}

function tologin(url){
	var ifs = document.getElementById("ifs");
	if(ifs){
		document.getElementById("ifs").src =url;
	}
}


function showsd(url){
	if(url==null||url==''){
		url=window.location.href;
	}
	window.open('/jsp_cn/shengda/call_sd.jsp?url='+url,'new','height='+600+',,innerHeight='+600+',width='+850+',innerWidth='+830+',top='+100+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

function openmsn(url){
	if(url==null||url==''){
		url=window.location.href;
	}
 	window.open('/jsp_cn/msn/callv5.jsp?url='+url,'new','height='+450+',,innerHeight='+450+',width='+500+',innerWidth='+500+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}
function setMSN_News(){
	var num = "0";
	var obj = document.getElementsByName('change');
	for(i=0;i<obj.length;i++){
		if(obj[i].checked){
			num = "1";
			break;
		}
	}
	if(num == "0"){
		alert("至少选择一项操作");
		return;
	}else{
		submitMSN_News();
	}
}
function submitMSN_News(){
	var url = "/jsp_cn/msn/update.jsp?time="+new Date().getTime();
	//var url = "/app/rr/aa?time="+new Date().getTime()+"&fn=setrr";

	if(document.getElementById("change_down").checked){
		url += "&cd=1";
	}
	if(document.getElementById("change_fav").checked){
		url += "&cf=1";
	}
	if(typeof(document.getElementById("change_bbs")) != 'undefined' && document.getElementById("change_bbs") != null && document.getElementById("change_bbs").checked){
		url += "&cb=1";
	}
	if(document.getElementById("change_message").checked){
		url += "&cm=1";
	}
	if(typeof(document.getElementById("change_doudan_fav")) != 'undefined' && document.getElementById("change_doudan_fav") != null && document.getElementById("change_doudan_fav").checked){
		url += "&cdf=1";
	}
	if(typeof(document.getElementById("change_doudan_top")) != 'undefined' && document.getElementById("change_doudan_top") != null && document.getElementById("change_doudan_top").checked){
		url += "&cdt=1";
	}

	jQuery("#setmsn").load(url);
}

function openqq(url){
	if(url==null||url==""){
		url=window.location.href;
	}
	window.open('/jsp_cn/qq/call_v2.jsp?url='+url,'new','height='+550+',,innerHeight='+550+',width='+600+',innerWidth='+600+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

function openrr(url){
	if(url==null || url==''){
		url=window.location.href;
	}
  	window.open('/jsp_cn/renren/call.jsp?url='+url,'new','height='+450+',,innerHeight='+450+',width='+800+',innerWidth='+800+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

function openlt(url){
	if(url==null || url==''){
		url=window.location.href;
	}
  	window.open('/jsp_cn/zglt/call.jsp?url='+url,'new','height='+700+',,innerHeight='+700+',width='+1300+',innerWidth='+1300+',top='+50+',left='+50+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

function openqq_app(url){
	if(url==null || url==''){
		url=window.location.href;
	}
	window.open('/jsp_cn/qq/call_v2.jsp?appflag=1&url='+url,'new','height='+400+',,innerHeight='+400+',width='+450+',innerWidth='+450+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');	
}
function setTXQQ_News(){
	var num = "0";
	var obj = document.getElementsByName('change');
	for(i=0;i<obj.length;i++){
		if(obj[i].checked){
			num = "1";
			break;
		}
	}
	if(num == "0"){
		alert("至少选择一项操作");
		return;
	}else{
		submitTXQQ_News();
	}
}
function submitTXQQ_News(){
		var url = "/jsp_cn/qq/update.jsp?time="+new Date().getTime();
		//var url = "/app/rr/aa?time="+new Date().getTime()+"&fn=setrr";
	
		if(document.getElementById("change_down").checked){
			url += "&cd=1";
		}
		if(document.getElementById("change_fav").checked){
			url += "&cf=1";
		}
		if(document.getElementById("change_message").checked){
			url += "&cm=1";
		}
		if(document.getElementById("change_top").checked){
			url += "&ct=1";
		}
		if(typeof(document.getElementById("change_doudan_fav")) != 'undefined' && document.getElementById("change_doudan_fav") != null && document.getElementById("change_doudan_fav").checked){
		url += "&cdf=1";
	}
	if(typeof(document.getElementById("change_doudan_top")) != 'undefined' && document.getElementById("change_doudan_top") != null && document.getElementById("change_doudan_top").checked){
		url += "&cdt=1";
	}
	
	jQuery("#settxqq").load(url);
}
function openmsn_iframe(url){
	if(url==""||url==null){
		url=window.location.href;
	}
 	window.open('/jsp_cn/msn/callv5.jsp?come=msn','new','height='+370+',,innerHeight='+370+',width='+453+',innerWidth='+453+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

function checkchrom(){
	parent.location.href='http://www.docin.com';
}
function checkchrom_showhidden(){
	parent.location.href='http://www.docin.com?check=1';
	
}
function showhiddenby_msniframe(){
	document.getElementById("newlogin").style.display="none";
	showsub();
}
 function opendx(url){
 	if(url==""||url==null){
		url=window.location.href;
	}
 	window.open('/jsp_cn/zgdx/call.jsp?url='+url,'new','height='+430+',,innerHeight='+430+',width='+500+',innerWidth='+500+',top='+150+',left='+150+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
 }
 function setZGDX_News(){
	var num = "0";
	var obj = document.getElementsByName('change');
	for(i=0;i<obj.length;i++){
		if(obj[i].checked){
			num = "1";
			break;
		}
	}
	if(num == "0"){
		alert("至少选择一项操作");
		return;
	}else{
		submitZGDX_News();
	}
}
function submitZGDX_News(){
	var url = "/jsp_cn/zgdx/update.jsp?time="+new Date().getTime();
	//var url = "/app/rr/aa?time="+new Date().getTime()+"&fn=setrr";

	if(document.getElementById("change_down").checked){
		url += "&cd=1";
	}
	if(document.getElementById("change_fav").checked){
		url += "&cf=1";
	}
	if(typeof(document.getElementById("change_bbs")) != 'undefined' && document.getElementById("change_bbs") != null && document.getElementById("change_bbs").checked){
		url += "&cb=1";
	}
	if(document.getElementById("change_message").checked){
		url += "&cm=1";
	}
	jQuery("#settxqq").load(url);
}

function openalipay(url){
	if(url==""||url==null){
		url=window.location.href;
	}
	window.open('/jsp_cn/alipay/call.jsp?url='+url,'new','height='+450+',,innerHeight='+450+',width='+950+',innerWidth='+950+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

function openzglt(){
	window.open('/jsp_cn/zglt/call.jsp','new','height='+600+',,innerHeight='+600+',width='+900+',innerWidth='+830+',top='+100+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}
function opentb(url){
	if(url==null || url==''){
		url=window.location.href;
	}
 	window.open('/jsp_cn/taobao/call.jsp?url='+url,'new','height='+550+',,innerHeight='+550+',width='+550+',innerWidth='+550+',top='+150+',left='+250+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}
function openkb(url){
	if(url==null || url==''){
		url=window.location.href;
	}
	window.open('/jsp_cn/kanbox/call.jsp?url='+url,'new','height='+550+',,innerHeight='+550+',width='+550+',innerWidth='+550+',top='+150+',left='+250+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}
function opentianyi(url){
	if(url==null || url==''){
		url=window.location.href;
	}
	window.open('/jsp_cn/tianyi/call.jsp?url='+url,'new','height='+550+',,innerHeight='+550+',width='+550+',innerWidth='+550+',top='+150+',left='+250+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}

function open360(url){
	if(url==null || url==''){
		url=window.location.href;
	}
	window.open('/jsp_cn/qihu360/call.jsp?url='+url,'new','height='+550+',,innerHeight='+550+',width='+550+',innerWidth='+550+',top='+150+',left='+250+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}
function loadnewlogin(){
	var newlogin= document.getElementById("newlogin");
	if(newlogin==null){
		jQuery("#loginwindow").load("/jsp_cn/weibo/login.jsp",function (data){
	});
	}
}
function openssv2(url){
	if(url==null || url==''){
		url=window.location.href;
	}
	window.open('/jsp_cn/weibo/callv2.jsp?url='+url,'new','height='+450+',,innerHeight='+450+',width='+550+',innerWidth='+550+',top='+200+',left='+200+',toolbar=no,menubar=no,scrollbars=auto,resizeable=no,location=no,status=no');
}


function titleNoticeYzmTitleD(){
	if(jQuery("#titleNoticeYzmTitleDiv").length>0){
			jQuery("#titleNoticeYzmTitleDiv").show();
	}
}

function refCodeLogin(){
	document.getElementById("titleloginimg").src = "/servlet/getimg?vfrom=loginFlase&t="+new Date().getTime();
}
function showLoginFailMes(){
	document.getElementById("noticeYzmTitleDiv").style.display = "block";
}
function showLoginCodeTitle(){
	var um = document.getElementById("username_new").value;
	jQuery.post("/app/my/docin/showCode4LoginFalse?um=" + encodeURIComponent(um) + "&t=" + new Date().getTime(), null, function(data){
		if(data != null && data != undefined && data == 1){
			var imgsrc = document.getElementById("titleloginimg").src;
			if(imgsrc == null || imgsrc.indexOf("getimg") == -1){
				document.getElementById("titleloginimg").src = "/servlet/getimg?vfrom=loginFlase&t=" + new Date().getTime();
			}
			document.getElementById("titleloginfailshowid").style.display = "block";
			document.getElementById("showcode4logintitle").value = 1;
		}else{
			document.getElementById("titleloginimg").src = "";
			document.getElementById("titleloginfailshowid").style.display = "none";
			document.getElementById("showcode4logintitle").value = 0;
		}
	});
}
function check_login_title(){
	var um = document.getElementById("username_new").value;
	jQuery.post("/app/my/docin/showCode4LoginFalse?um=" + encodeURIComponent(um) + "&t=" + new Date().getTime(), null, function(data){
		if(data != null && data != undefined && data == 1){
			var imgsrc = document.getElementById("titleloginimg").src;
			if(imgsrc == null || imgsrc.indexOf("getimg") == -1){
				document.getElementById("titleloginimg").src = "/servlet/getimg?vfrom=loginFlase&t=" + new Date().getTime();
			}
			document.getElementById("titleloginfailshowid").style.display = "block";
			document.getElementById("showcode4logintitle").value = 1;
			var u=document.loginnew.username.value;
			var p=document.loginnew.password.value;
			if(trim(u)=="" || trim(p)==""){
				alert("用户名/密码不能为空");
				return;
			}
			var reg = /[\u4E00-\u9FA5\uF900-\uFA2D]/;    
			if(reg.test(u)){
				alert("用户名只支持英文 数字的组合，请正确填写!");
				return;
			}
			var codevalue = document.getElementById("loginfalsecodetitle").value;
			if(rtrim(codevalue)==""){
				alert("对不起,请输入验证码！");
				document.getElementById('loginfalsecodetitle').select();
				return;
			}
			//document.getElementById("loginnew").submit();
			loginSubmitFunction();
		}else{
			document.getElementById("titleloginimg").src = "";
			document.getElementById("titleloginfailshowid").style.display = "none";
			document.getElementById("showcode4logintitle").value = 0;
			var u=document.loginnew.username.value;
			var p=document.loginnew.password.value;
			if(trim(u)=="" || trim(p)==""){
				alert("用户名/密码不能为空");
				return;
			}
			var reg = /[\u4E00-\u9FA5\uF900-\uFA2D]/;    
			if(reg.test(u)){
				alert("用户名只支持英文 数字的组合，请正确填写!");
				return;
			}
			//document.getElementById("loginnew").submit();
			loginSubmitFunction(); 
		}
	});
	
}

function loginSubmitFunction(){
	var um_tmp = jQuery("#username_new").val();
	var oPassVal_tmp = jQuery("#password_new").val();
	var showcode4login_tmp = jQuery("#showcode4logintitle").val();
	var loginfalsecode_tmp = jQuery("#loginfalsecodetitle").val();
	jQuery.post("/app/loginAjax/userLoginAjax.do?t=" + new Date().getTime(),{username:um_tmp,password:oPassVal_tmp,showcode4login:showcode4login_tmp,loginfalsecode:loginfalsecode_tmp},function(data){
		var flag = jQuery.trim((data.replace(/\r\n/gim,"")));
		if(flag == 1){
			jQuery("#loginnew").submit();
		}
		else if(flag == 2){
			alert("登录邮箱或密码错误！");
		}
		else if(flag == 3){
			alert("验证码错误");
		}
	});
}
