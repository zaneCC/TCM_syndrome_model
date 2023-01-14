var log_preNum = 0;
var toolHeight = 45;//46减去一像素边框
var scrollFlag = true;
var ft = 0;
var ftup = 0;
if(jQuery(".j_sidercontrol").length>0){
	var oRelaTop = jQuery(".j_sidercontrol").offset().top;
}
var relateDocH = 0;
function DocinReaderPlayer(){
	if(arguments.length==1){ 
		var arg = arguments[0];
	}
	this.config = {
		productId:0,
		allPage:0,
		previewNum:0,
		isBuy:1,
		baseUrl:"",
		flashUrl:picture_image_path_v1+"/players/PageViewer.swf?",
		initZoom:0,
		showStyle:1,
		contStyle:1,
		bookMark:0,
		isLogin:0,
		bookmarkid:-1,
		showAd:true,
		yjAdvStyle:'00',
		load_url:""
	};

	for(var p in arg){
		this.config[p] = arg[p];
	}
	this.iWidth = [792,1130];
	this.iHeight = [];
	this.metas={0:{width:720,height:540},1:{width:720,height:540},2:{width:720,height:540}};//记录文档高度
	this.screenWidth = window.screen.width;
	this.btn_zoomIn = jQuery("#j_zoomin");
	this.btn_zoomOut = jQuery("#j_zoomout");
	this.btn_comment = jQuery(".page_comment");
	this.btn_upload = jQuery(".page_upload");
	this.btn_search = jQuery(".page_search");
	this.btn_bookmark = jQuery(".book_mark");
	this.btn_fullscreen = jQuery(".btn_fullscreen");
	this.preBtn = jQuery("#preview"),
	this.nextBtn = jQuery("#nextPage");
	this.menu = jQuery('.doc_readmod,.reader_menu');
	this.topBtn = jQuery(".top_like");
	this.hitBtn = jQuery(".step_on");
	this.zoomFlag = 0;
	this.curPage = 1;
	this.myScale = 1;
	this.flashSize = {width:0,height:0};
	this.gotoPageInput = jQuery(".page_cur");
	this.floderBtn = jQuery("#floderbtn");
	this.adTj = {'1':[923504,923505,923506,923507,923508],'2':[939345,939347,939348,939349,939350]};
	//this.init();//是图片格式的时候直接初始化
	if(this.config.docstatus == 0){
		this.getMeta();//是flash格式的时候执行此方法
	}
	else{
		var screenH = jQuery(window).height(),
		pageOffsetTop = jQuery("#contentcontainer").offset().top;
		var pageNull = jQuery('<div id="pagenull" class="model panel" style="margin:0;"></div>').appendTo(jQuery("#contentcontainer"));
		jQuery("#pagenull").css({height:screenH-pageOffsetTop-50});
		jQuery("#floderbtn").hide();
		jQuery(".aside").css({marginTop:41});
		if(this.config.docstatus == 1){
			var ostatusP = jQuery('<p class="docStatusTips"><span class="txt">文档转换中</span></p>');
		}
		else if(this.config.docstatus == 2){
			var ostatusP = jQuery('<p class="docStatusTips"><span class="txt">文档审核中</span></p>');
		}
		else if(this.config.docstatus == 3){
			var ostatusP = jQuery('<p class="docStatusTips convertFailed"><span class="txt">该文档转化失败，<br/>请<a href="javascript:void(0);" onclick="downLoadNew('+this.config.productId+','+"1"+')">下载</a>后阅读。</span></p>');
		}
		ostatusP.appendTo(jQuery("#pagenull"));
		likeTooModControl(1);
	}
}
DocinReaderPlayer.prototype = {
	init:function(){
		var _this = this;
		_this.getMyScale();
		if(jQuery("#beforeAd").length>0){
			jQuery("#beforeAd").css({width:802}).show();
			if(!window.XMLHttpRequest){
				var oHeight = jQuery("#contentcontainer").height();
				jQuery("#beforeAd").css({height:oHeight});
			}
		}
		_this.createDiv();
		if(typeof(qwpart)!="undefined" && qwpart != 0){
			
			_this.gotoPages(qwpart,1);
		}
		_this.setBeforeAd();//前贴片
		_this.countHide(8,'endNum','beforeAd');//前贴片倒计时关闭
		if(typeof(isCartOrisCode)=="undefined"){
			_this.loadRelative();
		}
		if(showComment == "message"){
			_this.commentHandel();
		}
		_this.pageButton(_this.curPage);
		_this.zoomOut();
		_this.zoomIn();
		_this.fullscreen();
		_this.oSiderTop = jQuery(".aside").offset().top;
		_this.setPage(window);
		_this.initZoom();
		_this.prePage();
		_this.nextPage();
		_this.addBookMark();
		_this.jumpToPage();
		_this.readerMenu();
		_this.tabChangeHand();
		_this.initComment();
		_this.handMove();
		_this.hideSelected();
		if(typeof(isBuilding)!="undefined"){
			_this.setFloderBtn();//建筑新增代码
		}
		if(_this.config.showStyle == 2){
			jQuery(".doc_reader_mod").addClass('style_pic');
			jQuery("#maskLeft").show();
			jQuery("#maskRight").show();
			_this.cursorPage();
			if(jQuery("#j_hand").length>0&&jQuery("#j_select").length>0){
				jQuery("#j_hand").removeClass('drag_hand,btn_cur').addClass('drag_hand_no');
				jQuery("#j_select").removeClass('select_hand,btn_cur').addClass('select_hand_no');
			}
			if(typeof fromWebSearch !="undefined"){
				fromWebSearch();
			}
		}
		jQuery(".page_num").html('/'+_this.config.allPage);
		//如果书签存在的情况
		if(_this.config.bookMark>0 && qwpart == 0){
			_this.curPage = _this.config.bookMark;
			_this.gotoPage(_this.curPage,1);
			_this.btn_bookmark.addClass('book_mark_cur');
			jQuery("#bookmarktips p").html('你上次阅读到此页,点击这里返回第1页');
			jQuery("#bookmarktips").fadeIn().bind('click',function(){
				_this.gotoPage(1,1);
			});
			_this.setBookMark();
		}
		jQuery(document).bind('keydown',function(ev){
			if(ev.keyCode == 27&&_this.zoomFlag>0){
				_this.zoomInStyle(0);
			}
		});
		_this.floderBtn.bind('click',function(){
			if(jQuery(this).hasClass('btn_close')){
				_this.zoomInStyle(0);
				inpmv(5108);
			}
			else{
				jQuery(this).addClass('btn_close');
				_this.zoomInStyle(1);
				inpmv(5107);
			}
			_this.setBeforeAd();
		});
		//分享
		_this.btn_upload.bind('click',function(){
			_this.openShare(_this.config.productId);
		});
		if(jQuery(".new_share_more").length>0){
			jQuery(".new_share_more").bind("click",function(){
				_this.openShare(_this.config.productId);
			});
		}
		//付费阅读购买
		var oPayBtn = jQuery("#payForRead"),
			oGotoReadBtn = jQuery("#gotoRead");
		if(oGotoReadBtn.length>0){
			oGotoReadBtn.bind('click',function(){
				purchaseProduct(js_pid,tmp_userId,'buy');
			});
		}
		if(oPayBtn.length>0){
			oPayBtn.bind('click',function(){
				//if(_this.config.isLogin == 0){
					//showlogin();
					//return false;
				//}
				//purchasePayProduct(_this.config.productId,tmp_userId);//余额不足，弹出充值层
				clickPayRead(_this.config.productId,tmp_userId);
			});
		}
		_this.gotoPageInput.focusin(function(){
			jQuery(this).select();
		});
		jQuery("#bookmarktips .close").bind('click',function(event){
			jQuery("#bookmarktips").fadeOut();
			event.stopPropagation();
		});
	},
	tabInit:function(){
		var _this = this;
		_this.createDiv();
		if(_this.config.showStyle == 1){
			_this.setZoomFlashSize(_this.zoomFlag);
			if(_this.zoomFlag == 0){
				jQuery("#contentcontainer").css('paddingTop',0);
			}
			jQuery(".reader_container").css('height','auto');
			if(jQuery(".recommed_shadow").length>0&&jQuery(".recommedMod").length>0){
				jQuery(".recommed_shadow,.recommedMod").hide();
			}
		}
		_this.setPage(window);
		if(_this.config.showStyle == 2){
			_this.cursorPage();
			if(_this.zoomFlag == 0||_this.zoomFlag == 1){
				if(typeof fromWebSearch !="undefined"){
					fromWebSearch();
				}
			}
		}
		if(_this.zoomFlag == 1||_this.zoomFlag == 2||_this.zoomFlag ==3){
			setToolBarFix(2);
			siderFixedMove(2);
			jQuery("#contentcontainer").css('paddingTop',toolHeight);
		}
		jQuery(document).bind('keydown',function(ev){
			if(ev.keyCode == 27&&_this.zoomFlag>0){
				_this.zoomInStyle(0);
			}
		});
		log_preNum = 0;
		_this.gotoPage(_this.curPage,1);
		if(_this.config.showStyle == 1){
			jQuery(".scrollLoading").scrollLoading();
		}
		if((_this.zoomFlag == 2&&_this.config.showStyle == 2)||(_this.zoomFlag == 3&&_this.config.showStyle == 2)){
			if(jQuery(".backToTop").length>0){
				jQuery(".backToTop").hide();
			}
		}
		else{
			setBarPosition();
		}
		_this.zoomInStyle(_this.zoomFlag);
		_this.hideSelected();
	},
	commentHandel:function(){
		var _this = this;
		jQuery.ajax({
			url:'/jsp_cn/productEnd/load/doc_end_load_comment.jsp',
			data:'isLogin='+_this.config.isLogin,
			context: document.body,
			success:function(data){
				var oHtml = data;
				jQuery(this).append(oHtml);
				var cH = jQuery(window).height();
				var oCommentHeight = cH - cH*0.2 -142;
				jQuery("#showMsgContent").height(oCommentHeight);
				dialogBoxShadow();
				setObjCenter('j_comment');
				showMessageContent();
				var oTextArea = jQuery("#message");
				oYzm = jQuery("#checkout");
				if(oTextArea.length == 0){return;}
				oTextArea.bind('keydown',function(event){
					if(event.ctrlKey&&event.keyCode == 13){
						insertUserMessage(1);
					}
				});
				if(oYzm.length>0){
					oYzm.bind('keydown',function(event){
						if(event.ctrlKey&&event.keyCode == 13){
							insertUserMessage(1);
						}
					});
				}
				oTextArea.bind('focus',function(){
					showCheckCode();
				});
				jQuery('textarea[placeholder]').placeholder();
				oTextArea.focus();
			}
			});
	},
	initComment:function(){
		var _this = this;
		jQuery(".page_comment,#showComm").bind('click',function(){
			_this.commentHandel();
		});
	},
	openShare:function(pid){
		var popTmp = jQuery("#docinPopBox1");
			if(popTmp.length>0){
				dialogBoxShadow();
				setObjCenter('docinPopBox1');
				docinCopyData('docUrl','textUrl');
				docinCopyData('flashUrl','flUrl');
				docinCopyData('embedTxt','endShareCode');
			}
			/**shareEvent(pid,13);
			//add at 2013-04-25 分享弹窗请求
			jQuery.ajax({
				type:"POST",
				url:"/jsp_cn/productEnd/share_new_v1.jsp",
				context: document.body,
				data:"productid="+pid,
				async: false,
				success:function(re){
					if(re){
						jQuery(this).append(re);
						dialogBoxShadow();
						setObjCenter('docinPopBox1');
						docinCopyData('docUrl','textUrl');
						docinCopyData('flashUrl','flUrl');
						docinCopyData('embedTxt','endShareCode');
					}
				}
			});**/
	},
	clearFlash:function(pageno){
		var _this = this;
		_this.curPage = pageno;
		var aPages = jQuery("#contentcontainer .panel");
		jQuery.each(aPages,function(i){
			var tempI = i+1;
			if(tempI == pageno){return true;}
			if(tempI == pageno+1){return true;}
			if(tempI == pageno-1){return true;}
			jQuery(this).find("#flash_"+(i+1)).replaceWith('<div id="flash_'+(i+1)+'"></div>');
		});
	},
	addFlash:function(pageno){
		var _this = this;
		for(var p = pageno-1;p<=pageno+1;p++){
			if(p != 0&&jQuery("div#flash_"+p).length>0){
				var  flashvars={productId:readerConfig.productId,pagenum:p-1,loadurl:_this.config.load_url},params = {wmode:'transparent',allowscriptaccess:'always',hasPriority:'true'},attributes={};
				swfobject.embedSWF(_this.config.flashUrl, "flash_"+p, '100%', '100%', "9.0.0",picture_image_path_v1+'/js/expressInstall.swf',flashvars,params,attributes);
			}
		}
	},
	createDiv:function(){
		var _this = this;
		function createPageDiv(f){
			if(f == 1){
				var pageDiv = jQuery('<div id="page_'+i+'" class="model panel"><div class="panel_inner scrollLoading" data-url="'+_this.config.flashUrl+'"><div id="flash_'+i+'"></div></div></div>').appendTo(jQuery("#contentcontainer"));
					jQuery("#page_"+i+" .panel_inner").css({width:_this.setFlashSize(0,i)[0],height:_this.setFlashSize(0,i)[1]});
			}
			else if(f == 2){
				var pageDiv = jQuery('<div style="width:0px;height:0px;" id="page_'+i+'" class="model panel"><div class="panel_inner"><div id="flash_'+i+'"></div></div></div>').appendTo(jQuery("#contentcontainer"));
			}
		}
		if(_this.config.showStyle == 1){//1：竖版 2：横版
			if(_this.config.isBuy == 1){//已付费
				jQuery("#contentcontainer").html("");
				var createNum = _this.config.allPage;
				for(var i = 1;i<=createNum;i++){
					createPageDiv(1);
				}
				jQuery('<div id="moveHandel"></div>').appendTo(jQuery("#contentcontainer"));
			}
			else if(_this.config.isBuy == 0){//未付费，只加载预览页数
				jQuery("#contentcontainer").html("");
				var createNum = _this.config.previewNum;
				for(var i = 1;i<=createNum;i++){
					createPageDiv(1);
				}
				jQuery('<div id="moveHandel"></div>').appendTo(jQuery("#contentcontainer"));
			}
			else if(_this.config.isBuy == 2){//付费或下载完 加载付费页数
				for(var i =_this.config.previewNum+1;i<=_this.config.allPage;i++){
					createPageDiv(1);
				}
				_this.config.isBuy = 1;
			}
			if(jQuery(".preview_end").length>0){
				jQuery(".preview_end").show(); 
			}
			if(_this.config.showAd == true){
				if(_this.config.yjAdvStyle == '0'){
					_this.insertAd(2,1);//A样式
					//统计 s1
				}
				else if(_this.config.yjAdvStyle == '1'){
					//_this.insertAd(2,2);
					//统计 s2
					_this.insertAd(2);
				}
			}
			jQuery(".scrollLoading").scrollLoading();
			likeTooModControl(1);
		}
		else if(_this.config.showStyle == 2){
			if(_this.config.isBuy == 1){
				jQuery("#contentcontainer").html("");
				var createNum = _this.config.allPage;
				for(var i = 1;i<=createNum;i++){
					createPageDiv(2);
				}
				jQuery("#page_1").removeAttr('style');
			}
			else if(_this.config.isBuy == 0){
				jQuery("#contentcontainer").html("");
				var createNum = _this.config.previewNum;
				for(var i = 1;i<=createNum;i++){
					createPageDiv(2)
				}
				jQuery("#page_1").removeAttr('style');
			}
			else if(_this.config.isBuy == 2){
				for(var i = _this.config.previewNum+1;i<=_this.config.allPage;i++){
					createPageDiv(2)
				}
				jQuery("#page_"+_this.curPage).removeAttr('style');
				_this.config.isBuy = 1;
			}
			_this.setReaderHeight();
			_this.addFlash(_this.curPage);
			_this.clearFlash(_this.curPage);
			likeTooModControl(1);
		}
	},
	setFlashSize:function(iZoom,pageno){
		var _this = this;
		var iHeight = parseInt(_this.iWidth[iZoom]/_this.myScale);
		_this.iHeight[iZoom] = iHeight;
		return [_this.iWidth[iZoom],iHeight];
	},
	setZoomFlashSize:function(iZoom){
		var _this = this;
		var aFlash = jQuery("#contentcontainer .panel_inner");

		if(_this.screenWidth<=1280){
			_this.iWidth[2] = 1240;	
			_this.iWidth[3] = 1695;	
		}
		else if(_this.screenWidth>1280){
			_this.iWidth[2] = _this.screenWidth-40;
			_this.iWidth[3] = _this.screenWidth*1.5;
		}
		else if(_this.screenWidth>2560){
			_this.iWidth[2] = 2520;
			_this.iWidth[3] = 3840;
		}
		jQuery.each(aFlash,function(i){
			var zoomHeight = parseInt(_this.iWidth[iZoom]/_this.myScale);
			_this.iHeight[iZoom] = zoomHeight;
			jQuery(this).css({width:_this.iWidth[iZoom],height:zoomHeight});
		});
		if(iZoom == 3){
			var sl = (_this.iWidth[3]-jQuery(window).width())/2;
			jQuery(window).scrollLeft(sl);
		}
	},
	getMyScale:function(){
		var _this = this;
		//var pageIndex = parseInt(pageno/50);
		var pageIndex = 0;
		_this.myScale =_this.getScale(_this.metas[pageIndex].width,_this.metas[pageIndex].height);
	},
	getScale:function(w,h){
		var _this = this;
		var iScale = (w/h).toFixed(4);
		return iScale;
	},
	setReaderHeight:function(){
		var _this = this;
		var st = jQuery(window).scrollTop();
		var oReaderTop = Math.ceil(jQuery(".reader_container").offset().top),
			oScreenHeight = jQuery(window).height(),
			oToolHeight = jQuery(".docin_reader_tools").outerHeight();
			if(oReaderTop>206){oReaderTop = 206;}
			if(_this.zoomFlag == 0){
				var cH = oScreenHeight-oReaderTop;
				var docHeight = cH-2;
			}
			if(_this.zoomFlag>0){
				var cH = oScreenHeight;
				var docHeight = cH - oToolHeight-2;
			}
			if(docHeight<390){
				docHeight = 420;
			}
			var cW = jQuery("#contentcontainer").width();
			var tempWidth =Math.ceil(docHeight*_this.myScale);//如果适高
			var tempHeight =Math.ceil(cW/_this.myScale);//如果适宽
			if(tempWidth>cW){//按照宽来
				if(_this.zoomFlag == 0){
					jQuery(".reader_container").css('height',tempHeight);
					jQuery(".panel_inner").css({width:cW,height:tempHeight});
				}
				else if(_this.zoomFlag == 1){
					jQuery(".reader_container").css('height',tempHeight);
					jQuery(".panel_inner").css({width:cW,height:tempHeight});
					jQuery("#contentcontainer").css('paddingTop',oToolHeight);
				}
				else{
					jQuery(".reader_container").css('height',cH-2);
					jQuery(".panel_inner").css({width:cW,height:tempHeight});
					var mart = Math.floor(cH - tempHeight)/2;
					jQuery("#contentcontainer").css('paddingTop',mart);
				}
			}
			else{//按照高来
				jQuery(".panel_inner").css({width:tempWidth,height:docHeight});
				if(_this.zoomFlag == 0){
					jQuery(".reader_container").css('height',docHeight);
					jQuery("#contentcontainer").css('paddingTop',0);
				}
				else if(_this.zoomFlag>0){
					jQuery(".reader_container").css('height',cH-2);
					jQuery("#contentcontainer").css('paddingTop',oToolHeight);
				}
			}
	},
	cursorPage:function(){
		//判断鼠标的位置
		var _this = this;
		var oMaskLeft = jQuery("#maskLeft"),
			oMaskRight = jQuery("#maskRight");

		oMaskLeft.bind('mouseover',function(event){
			oMaskLeft.css({cursor:'url('+picture_image_path_v1+'/images_cn/news/html_reader/arr_left.cur),auto'});
		});
		oMaskRight.bind('mouseover',function(event){
			oMaskRight.css({cursor:'url('+picture_image_path_v1+'/images_cn/news/html_reader/arr_right.cur),auto'});
		});
		oMaskLeft.unbind('click');
		oMaskLeft.bind('click',function(){
			_this.preBtn.triggerHandler("click");
		});
		oMaskRight.unbind('click');
		oMaskRight.bind('click',function(){
			_this.nextBtn.triggerHandler("click");
		});
		jQuery(document).unbind("keydown");
		jQuery(document).bind("keydown",function(event){
			if(_this.config.showStyle == 2){
				_this.tabKeyPage(event);
			}
		});
	},
	tabKeyPage:function(event){
		var _this = this;
		var oKey = event.keyCode; 
		if(oKey == 37){
			_this.preBtn.triggerHandler("click");
			inpmv(5464);
			return;
		}
		if(oKey == 39){
			_this.nextBtn.triggerHandler("click");
			inpmv(5466);
			return;
		}
	},
	getMeta:function(){
		var _this = this;
		var aJaxNum = Math.ceil(_this.config.allPage/50);
		aJaxNum = 1;
		var metaBaseUrl = 'http://page.douding.cn/docinfile2/meta_'+_this.config.productId+'_';
		for(var i = 0;i<aJaxNum;i++){
			jQuery.ajax({
				type:'get',
				url:metaBaseUrl+i+'.docin',
				dataType: 'jsonp',
				success:function(data){
					_this.metas[0] = data;
					_this.init();

				},
				error:function(){
					//倒霉，没有获取到
				}
			});
		}
	},
	initZoom:function(){
		var _this = this;
		if(this.config.initZoom ==1){
			_this.zoomInStyle(1);
		}
		else if(this.config.initZoom ==2){
			_this.zoomInStyle(2);
		}
		else if(this.config.initZoom ==0){
			_this.zoomInStyle(0);
		}
	},
	fullscreen:function(){
		var _this = this;
		_this.btn_fullscreen.bind('click',function(){
			if(jQuery(this).hasClass('q')){//正在全屏,点击退出
				_this.zoomInStyle(0);
				inpmv(5118);
			}
			else{
				_this.zoomInStyle(2);
				_this.floderBtn.hide();
				inpmv(5070);
			}
			
		});
	},
	escapeFullScreen:function(){
		var _this = this;
		_this.zoomFlag = 0;
		jQuery(".page_body").removeClass('full_2');
		jQuery(".page_body").removeClass('full_3');//退出大全屏
		jQuery(".main").removeClass('full_1');
		_this.btn_zoomOut.addClass('zoom_out_no').removeClass('zoom_out');
		_this.controlSomeDiv(1);
		_this.btn_fullscreen.removeClass('q');
		_this.btn_fullscreen.html('全屏');
		_this.btn_zoomIn.removeClass('zoom_in_no').addClass('zoom_in');
		if(_this.config.showStyle == 1){
			_this.setZoomFlashSize(0);
		}
		if(_this.config.showStyle == 2){
			_this.setReaderHeight();
		}
		_this.gotoPage(_this.curPage);
		if(jQuery(".w100").length>0){
			jQuery(".w100").show();
		}
		_this.floderBtn.show().removeClass('btn_close');
	},
	zoomOut:function(){
		var _this = this;
		if(_this.btn_zoomOut.length>0){
			_this.btn_zoomOut.bind('click',function(){
				_this.btn_fullscreen.html('全屏');
				if(_this.zoomFlag == 3){
					jQuery(".page_body").removeClass('full_3');
					_this.zoomInStyle(2);
				}
				else if(_this.zoomFlag == 2){
					_this.zoomInStyle(1);
				}
				else if(_this.zoomFlag == 1){
					_this.floderBtn.show().removeClass('btn_close');
					_this.zoomInStyle(0);
				}
				else if(_this.zoomFlag == 0){
					return false;
				}
								
			});
		}
	},
	zoomIn:function(){
		var _this = this;
		if(_this.btn_zoomIn.length>0){
			_this.btn_zoomIn.bind('click',function(){
				if(_this.zoomFlag == 0){
					_this.zoomInStyle(1);
				}
				else if(_this.zoomFlag == 1){
					_this.zoomInStyle(2);
					_this.floderBtn.hide();
				}
				else if(_this.zoomFlag == 2){
					_this.zoomInStyle(3);
					_this.floderBtn.hide();
				}
			});
		}
	},
	zoomInStyle:function(f){
		var _this = this;
		if(_this.zoomFlag == 0&&jQuery(window).scrollTop()<=_this.oSiderTop){
			_this.temoScale = (jQuery(window).scrollTop()+(jQuery(window).height()/3)+166)/jQuery(document).height();
		}else{
			_this.temoScale = (jQuery(window).scrollTop()+(jQuery(window).height()/3))/jQuery(document).height();
		}
		if(f == 0){
			jQuery(".page_wrap").css({marginTop:50});
			if(!window.XMLHttpRequest){jQuery(".page_wrap").css({marginTop:0});}
			_this.escapeFullScreen();
			if(jQuery(".w100").length>0){
				jQuery(".w100").show();
			}
			setToolBarFix(1);
			siderFixedMove(1);
			if(_this.config.showStyle == 2){
				jQuery(".main").css({width:802});
				_this.setReaderHeight();
			}
			likeTooModControl(1);
			_this.btn_fullscreen.html('全屏').removeClass('q').data("label",'全屏');
			if(_this.config.showStyle == 1){
				jQuery(".main").css({width:802});
				jQuery("#contentcontainer").css({paddingTop:0});
				if(typeof(isBuilding)!="undefined"){
				_this.relativeControl();//建筑终极页新增代码
				}
			}
			if(typeof(isBuilding)!="undefined"){
				_this.setFloderBtn();//建筑新增代码
			}
			_this.gotoPage(_this.curPage);
			setBarPosition();
			controlLeftBottomAdv(1);
			return;
		}
		jQuery(".page_wrap").css({marginTop:0});
		_this.controlSomeDiv(0);//隐藏不需要的div
		if(f == 1){
			_this.zoomFlag = 1;
			jQuery(".main").addClass('full_1');
			if(_this.config.showStyle == 1){
				jQuery(".full_1").css({width:_this.iWidth[1]});
			}
			_this.btn_zoomOut.removeClass('zoom_out_no').addClass('zoom_out');
			_this.btn_zoomIn.removeClass('zoom_in_no').addClass('zoom_in');
			_this.floderBtn.addClass('btn_close').show();
			_this.btn_fullscreen.html('全屏').removeClass('q').data("label",'全屏');
			jQuery(".page_body").removeClass('full_2');
			jQuery(".page_body").removeClass('full_3');
			if(_this.config.showStyle == 2){
				setBarPosition();
				jQuery(".main").css({width:'100%'});
			}
			likeTooModControl(1);
			controlLeftBottomAdv(2);
		}
		if(f == 2){
			_this.zoomFlag = 2;
			jQuery(".main").addClass('full_1');
			jQuery(".page_body").addClass('full_2');
			if(_this.config.showStyle == 1){
				jQuery(".full_1").css({width:_this.iWidth[2]});
			}
			if(_this.config.showStyle == 2){
				jQuery(".main").css({width:'100%'});
			}
			_this.btn_zoomIn.removeClass('zoom_in_no').addClass('zoom_in');
			_this.btn_zoomOut.removeClass('zoom_out_no').addClass('zoom_out');
			_this.floderBtn.hide();
			_this.btn_fullscreen.html('取消全屏').addClass('q').data("label",'取消全屏');;
			likeTooModControl(2);
		}
		if(f == 3){
			_this.zoomFlag = 3;
			jQuery(".main").addClass('full_1');
			jQuery(".page_body").addClass('full_3');
			if(_this.config.showStyle == 1){
				jQuery(".full_1").css({width:_this.iWidth[3]});
			}
			if(_this.config.showStyle == 2){
				jQuery(".main").css({width:'100%'});
			}
			_this.floderBtn.hide();
			_this.btn_zoomIn.removeClass('zoom_in').addClass('zoom_in_no');
			_this.btn_zoomOut.removeClass('zoom_out_no').addClass('zoom_out');
			likeTooModControl(2);
		}
		if(_this.config.showStyle == 2){
			_this.setReaderHeight();
			if(f == 2||f == 3){
				if(jQuery(".backToTop").length>0){
					jQuery(".backToTop").hide();
				}
			}
		}
		else{
			setBarPosition();
		}
		jQuery("#contentcontainer").css('paddingTop',toolHeight);
		if(_this.config.showStyle == 1){
			_this.setZoomFlashSize(f);
		}
		setToolBarFix(2);
		siderFixedMove(2);
		_this.gotoPage(_this.curPage);
		if(jQuery(".w100").length>0){
			jQuery(".w100").hide();
		}
		if(typeof(isBuilding)!="undefined"){
			_this.setFloderBtn();//建筑新增代码
		}
	},
	controlSomeDiv:function(f){
		var _this = this;
		var aDiv = ['user_doc_mod','.page_crubms','.doc_header_mod','.relative_doc_mod','.recent_cart_mod','.recent_doc_mod','.doc_hd_mini','.today_free','.sider_guanggao','.sider_guanggao_2','.col_box','.j_sidercontrol_build'];
		if(f == 0){
			jQuery.each(aDiv,function(i){
				if(jQuery(aDiv[i]).length>0){
					jQuery(aDiv[i]).hide();
				}
			})
		}
		if(f == 1){
			jQuery.each(aDiv,function(i){
				if(jQuery(aDiv[i]).length>0){
					jQuery(aDiv[i]).show();
				}
			})
		}
	},
	prePage:function(){
		var _this = this;
		_this.preBtn.bind('click',function(){
			if(_this.config.showStyle == 2){
				if(_this.flagPage == _this.config.allPage+1){
					_this.curPage = _this.flagPage;
					_this.flagPage = 0;
					if(jQuery(".recommed_shadow").length>0){
						jQuery(".recommed_shadow").hide();
					}
					if(jQuery(".recommedMod").length>0){
						jQuery(".recommedMod").hide();
					}
				}
			}
			_this.curPage --;
			if(_this.curPage<1){
				_this.curPage = 1;
				return;
			}
			_this.gotoPage(_this.curPage,2);
			_this.pageButton(_this.curPage);

		});
	},
	nextPage:function(){
		var _this = this;
		_this.nextBtn.bind('click',function(){
			_this.curPage ++;
			if(_this.curPage>_this.config.allPage){
				_this.curPage = _this.config.allPage;
				if(_this.config.showStyle == 2){
					_this.flagPage = _this.config.allPage+1;
					if(jQuery('.recommedMod').length>0){
						jQuery(".recommed_shadow").show();
						jQuery(".recommedMod").show();
						if(!window.XMLHttpRequest){
							jQuery(".recommed_shadow").height(jQuery("#contentcontainer").height());
						}
					}
					else{
						jQuery.ajax({
							url:'/app/p/end/ajax/getLastPageRecommends.do',
							data:'productId='+_this.config.productId,
							context:jQuery(".doc_reader_mod"),
							success:function(data){
								if(jQuery('.recommedMod').length>0&&jQuery(".recommed_shadow").length>0){
									return;
								}
								var oHtml = data;
								var oShadow = jQuery('<div class="recommed_shadow"></div>').appendTo(jQuery(this));
								jQuery(this).append(oHtml);
								jQuery("#jReview").bind("click",function(){
									_this.gotoPage(1);
									_this.pageButton(1);
									_this.curPage = 1;
									_this.flagPage = 0;
									jQuery(".recommed_shadow").hide();
									jQuery(".recommedMod").hide();
								});
								jQuery("#jCloseRecomm").bind("click",function(){
									_this.preBtn.click();
								});
								if(!window.XMLHttpRequest){
									jQuery(".recommed_shadow").height(jQuery("#contentcontainer").height());
								}
							}
						});
					}
				}
				return;}
			_this.gotoPage(_this.curPage,2);
			_this.pageButton(_this.curPage);
		});
	},
	gotoPages:function(pageno,k){
		var _this = this;
		if(_this.config.showStyle == 1){
			if(_this.config.isBuy == 0){
				if(pageno>_this.config.previewNum){
					_this.gotoPageInput.val(_this.config.previewNum);
					jQuery(window).scrollTop(jQuery(document.body).height());
					return false;
				}
			}
			if(jQuery("#page_"+pageno).length>0){
					//var preTop = jQuery("#page_"+(pageno-1)).offset().top;
					//var curObjTop = preTop+_this.iHeight[_this.zoomFlag]-54;
				if(pageno == 1){
					var preTop = jQuery("#page_"+pageno).offset().top;
					var curObjTop = 0;
				}
				else if(pageno == 2){
					if(jQuery(window).scrollTop()>_this.oSiderTop){
						var curObjTop = jQuery("#page_"+pageno).offset().top-44;
					}
					else{
						var curObjTop = jQuery("#page_"+pageno).offset().top-37-44;
					}
				}
				else{
					var preTop = jQuery("#page_"+pageno).offset().top-37-44;
					var curObjTop = preTop;
				}
					
			}
			if(k && k == 1){
				jQuery(window).scrollTop(curObjTop);
			}
			else if(k && k ==2){
				jQuery("html,body").stop().animate({ scrollTop:curObjTop}, 200);
			}
			else{
				if(_this.zoomFlag == 0&&jQuery(window).scrollTop()<=_this.oSiderTop){
					var ttt = _this.temoScale*jQuery(document).height()-jQuery(window).height()/3-166;
				}
				else{
					var ttt = _this.temoScale*jQuery(document).height()-jQuery(window).height()/3;
				}
				jQuery(window).scrollTop(ttt);
			}
		}
		if(_this.config.showStyle == 2){
			_this.curPage = pageno;
			if(_this.config.isBuy == 0){
				if(pageno>_this.config.previewNum){
					_this.gotoPageInput.val(_this.config.previewNum+1);
					_this.curPage = _this.config.previewNum+1;
					if(jQuery(".preview_end").length>0){
						jQuery(".model").css({width:0,height:0});
						jQuery(".preview_end").show();
					}
				}
				else{
					if(jQuery(".preview_end").length>0){
						jQuery(".preview_end").hide();
					}
					_this.gotoPageInput.val(_this.curPage);
					_this.addFlash(pageno);
					_this.clearFlash(pageno);
					jQuery(".model").css({width:0,height:0});
					jQuery("#page_"+pageno).removeAttr('style');
					_this.bookMarkShow(pageno);
					if(!window.XMLHttpRequest){
						var w = jQuery("#page_"+pageno).find(".panel_inner").width();
						var h = jQuery("#page_"+pageno).find(".panel_inner").height();
						jQuery("#page_"+pageno).css({width:w,height:h});
					}
				}
			}
			else{
				_this.gotoPageInput.val(_this.curPage);
				_this.addFlash(pageno);
				_this.clearFlash(pageno);
				jQuery(".model").css({width:0,height:0});
				jQuery("#page_"+pageno).removeAttr('style');
				_this.bookMarkShow(pageno);
				if(!window.XMLHttpRequest){
						var w = jQuery("#page_"+pageno).find(".panel_inner").width();
						var h = jQuery("#page_"+pageno).find(".panel_inner").height();
						jQuery("#page_"+pageno).css({width:w,height:h});
					}
			}
		}	
	},
	gotoPage:function(pageno,k){
		var _this = this;
		if(_this.config.showStyle == 1){
			if(_this.config.isBuy == 0){
				if(pageno>_this.config.previewNum){
					_this.gotoPageInput.val(_this.config.previewNum);
					jQuery(window).scrollTop(jQuery(document.body).height());
					return false;
				}
			}
			if(jQuery("#page_"+pageno).length>0){
					//var preTop = jQuery("#page_"+(pageno-1)).offset().top;
					//var curObjTop = preTop+_this.iHeight[_this.zoomFlag]-54;
				if(pageno == 1){
					var preTop = jQuery("#page_"+pageno).offset().top;
					var curObjTop = 0;
				}
				else if(pageno == 2){
					if(jQuery(window).scrollTop()>_this.oSiderTop){
						var curObjTop = jQuery("#page_"+pageno).offset().top-44;
					}
					else{
						var curObjTop = jQuery("#page_"+pageno).offset().top-37-44;
					}
				}
				else{
					var preTop = jQuery("#page_"+pageno).offset().top-44;
					var curObjTop = preTop;
				}
					
			}
			if(k && k == 1){
				jQuery(window).scrollTop(curObjTop);
			}
			else if(k && k ==2){
				jQuery("html,body").stop().animate({ scrollTop:curObjTop}, 200);
			}
			else{
				if(_this.zoomFlag == 0&&jQuery(window).scrollTop()<=_this.oSiderTop){
					var ttt = _this.temoScale*jQuery(document).height()-jQuery(window).height()/3-166;
				}
				else{
					var ttt = _this.temoScale*jQuery(document).height()-jQuery(window).height()/3;
				}
				jQuery(window).scrollTop(ttt);
			}
		}
		if(_this.config.showStyle == 2){
			_this.curPage = pageno;
			if(_this.config.isBuy == 0){
				if(pageno>_this.config.previewNum){
					_this.gotoPageInput.val(_this.config.previewNum+1);
					_this.curPage = _this.config.previewNum+1;
					if(jQuery(".preview_end").length>0){
						jQuery(".model").css({width:0,height:0});
						jQuery(".preview_end").show();
					}
				}
				else{
					if(jQuery(".preview_end").length>0){
						jQuery(".preview_end").hide();
					}
					_this.gotoPageInput.val(_this.curPage);
					_this.addFlash(pageno);
					_this.clearFlash(pageno);
					jQuery(".model").css({width:0,height:0});
					jQuery("#page_"+pageno).removeAttr('style');
					_this.bookMarkShow(pageno);
					if(!window.XMLHttpRequest){
						var w = jQuery("#page_"+pageno).find(".panel_inner").width();
						var h = jQuery("#page_"+pageno).find(".panel_inner").height();
						jQuery("#page_"+pageno).css({width:w,height:h});
					}
				}
			}
			else{
				_this.gotoPageInput.val(_this.curPage);
				_this.addFlash(pageno);
				_this.clearFlash(pageno);
				jQuery(".model").css({width:0,height:0});
				jQuery("#page_"+pageno).removeAttr('style');
				_this.bookMarkShow(pageno);
				if(!window.XMLHttpRequest){
						var w = jQuery("#page_"+pageno).find(".panel_inner").width();
						var h = jQuery("#page_"+pageno).find(".panel_inner").height();
						jQuery("#page_"+pageno).css({width:w,height:h});
					}
			}
		}	
	},
	pageButton:function(pageno){
		var _this = this;
		if(pageno){
			if(_this.config.allPage == 1){
				_this.preBtn.removeClass('page_prev').addClass('page_prev_no').removeAttr('title');
				_this.nextBtn.addClass('page_next').addClass('page_next_no').removeAttr('title');

			}
			else if(pageno == 1){
				_this.preBtn.removeClass('page_prev').addClass('page_prev_no').removeAttr('title');
				_this.nextBtn.addClass('page_next').removeClass('page_next_no');
			}
			else if(pageno == _this.config.allPage){
				_this.nextBtn.removeClass('page_next').addClass('page_next_no').removeAttr('title');
				_this.preBtn.addClass('page_prev').removeClass('page_prev_no');
			}
			else{
				_this.preBtn.addClass('page_prev').removeClass('page_prev_no');
				_this.nextBtn.addClass('page_next').removeClass('page_next_no');
			}
		}
	},
	setBookMark:function(){
		var oTimer = setTimeout(function(){
				jQuery("#bookmarktips").fadeOut();
			},3000);
			jQuery("#bookmarktips").bind('mouseover',function(){
				clearTimeout(oTimer);
			});
			jQuery("#bookmarktips").bind('mouseout',function(){
				oTimer = setTimeout(function(){
					jQuery("#bookmarktips").fadeOut();
				},3000);
		});
	},
	addBookMark:function(){
		var _this = this;
		_this.btn_bookmark.bind('click',function(){
			if(jQuery(this).hasClass('book_mark_cur')){//取消标签
				jQuery.post("/app/my/bookself/delbookmark.do",{productid:_this.config.productId,bookmarkid:_this.config.bookmarkid,fromdomain:""},function(data){
					if(data == 1){
						_this.config.bookMark = 0;
						_this.btn_bookmark.removeClass('book_mark_cur');
						jQuery("#bookmarktips p").html('书签已取消');
						jQuery("#bookmarktips").unbind("click").fadeIn();
						_this.setBookMark();
					}
				});
			}
			else{
				if(_this.config.isLogin == 0){
					showlogin();
				}
				else{
					if(isZuZhi == "1"){
						mailDialogAlert(login_email,landingEmail,domailUrl);
						return;
					}
					jQuery.post("/app/my/bookself/addbookmark.do",{productid:_this.config.productId,bookmarkid:_this.config.bookmarkid,bookmark:_this.curPage,fromdomain:""},function(data){
					_this.config.bookMark = _this.curPage;
					readerConfig.bookMark = _this.curPage;
					_this.btn_bookmark.addClass('book_mark_cur');
					jQuery("#bookmarktips p").html('添加书签成功,可以进入我的书房中查看。<a title="点击查看" target="_blank" href="http://shufang.docin.com/">点击查看&gt;&gt;</a>');
					jQuery("#bookmarktips").unbind("click").fadeIn();
					jQuery("#addBookShop").removeClass('green').addClass('gray').html('<span class="ico_mini ico_addfav"></span>已收藏');
					_this.setBookMark();
					//暂时不知道返回值什么用途 估计没用
					});
					
				}
			}
		});
	},
	bookMarkShow:function(pageno){
		var _this = this;
		if(pageno == _this.config.bookMark){
			_this.btn_bookmark.addClass('book_mark_cur');
		}
		else{
			_this.btn_bookmark.removeClass('book_mark_cur');
		}
	},
	jumpToPage:function(){
		var _this = this;
		_this.gotoPageInput.bind('keydown',function(ev){
			if(ev.keyCode == 13){
				var curpage = parseInt(jQuery(this).val());
				if(isNaN(curpage)){
					return false;
				}
				if(curpage<=0||curpage>_this.config.allPage){
					return false;
				}
				_this.curPage = curpage;
				_this.pageButton(_this.curPage);
				_this.gotoPage(_this.curPage,1);
				if(jQuery('.recommedMod').length>0&&_this.config.showStyle == 2){
					jQuery(".recommed_shadow").hide();
					jQuery(".recommedMod").hide();
				}
			}
		});
	},
	readerMenu:function(){
		var _this = this;
		var menuList = jQuery(".menu_list_wrap");
		var otime = null;
		if(menuList.length<0){return false;}
		_this.menu.bind('mouseover',function(){
			clearTimeout(otime);
			menuList.show();
		}).bind('mouseleave',function(){
			otime = setTimeout(function(){
				menuList.hide();
			},100);
			
		});
		//菜单功能
		var oShowInfo = jQuery("#j_showinfo"),
		oShowAbout = jQuery("#j_about"),
		oJlist = jQuery("#j_list"),
		oJpic = jQuery("#j_pic");
		if(oShowInfo.length>0){
			oShowInfo.bind('click',function(){
				var docInfo = new CreateDocinDialog(showInfoConfig);
			});
		}
		if(oShowAbout.length>0){
			oShowAbout.bind('click',function(){
				var docAbout = new CreateDocinDialog({
					id:'docAbout',
					title:'关于豆丁',
					button:2,
					shadow:1,
					content:'<span class="fwb">豆丁网是中国最领先的C2C文档分享网站。</span><br/>豆丁网(www.docin.com)创立于2008年，致力于打造全球最大的中文社会化阅读平台，为用户提供一切有价值的可阅读之物。经过几年的运营，豆丁网已经成功跻身互联网全球500强，成为提供垂直服务的优秀网站之一。目前豆丁网拥有分类广泛的实用文档、众多出版物、行业研究报告、以及数千位行业名人贡献的专业文件，各类读物总数超过两亿，是目前全球最大的中文文档库。'
				});
			});
		}
		if(oJlist.length>0){
			oJlist.bind('click',function(){
				if(jQuery(this).hasClass("cur")){return;}
				jQuery(window).unbind('resize');
				jQuery(window).unbind('scroll');
				jQuery("#contentcontainer").html("");
				_this.config.showStyle = 1;
				jQuery(".doc_reader_mod").removeClass('style_pic');
				jQuery("#maskLeft").hide();
				jQuery("#maskRight").hide();
				_this.tabInit();
				//列表模式下手型和复制放开使用
				if(jQuery("#j_hand").length>0&&jQuery("#j_select").length>0){
					jQuery("#j_hand").removeClass('drag_hand_no');
					jQuery("#j_select").removeClass('select_hand_no');
				}
				if(jQuery(".preview_end").length>0){
					jQuery(".preview_end").show();
				}
				jQuery(".menu_list li").removeClass('cur');
				jQuery(this).addClass('cur');
				inpmv(5114);
			});
		}
		if(oJpic.length>0){
			oJpic.bind('click',function(){
				if(jQuery(this).hasClass("cur")){return;}
				jQuery(window).unbind('resize');
				jQuery(window).unbind('scroll');
				jQuery("#contentcontainer").html("");
				_this.config.showStyle = 2;
					if(_this.config.showAd){
						jQuery(".relative_doc_inner").height("auto");	
					}
					if(jQuery(".preview_end").length>0){
						jQuery(".preview_end").hide();
					}
					jQuery(".doc_reader_mod").addClass('style_pic');
					jQuery("#maskLeft").show();
					jQuery("#maskRight").show();
					_this.tabInit();
					//ppt模式下手型和复制都无法使用
					if(jQuery("#j_hand").length>0&&jQuery("#j_select").length>0){
						jQuery("#j_hand").addClass('drag_hand_no');
						jQuery("#j_select").addClass('select_hand_no');
					}
					jQuery(".menu_list li").removeClass('cur');
					jQuery(this).addClass('cur');
					inpmv(5115);
			});
		}
	}, 
	insertAd:function(f){
		var _this = this;
		var readerAd1 = jQuery('<div id="ad1" class="adBox"><iframe src="/jsp_cn/ad/end_page.jsp?loc=1" width="760" height="90" scrolling="no" frameborder="0" style="display:block;margin:0 auto;"></iframe></div>');
		var readerAd2 = jQuery('<div id="ad2" class="adBox"><iframe src="/jsp_cn/ad/end_page.jsp?loc=2" width="760" height="90" scrolling="no" frameborder="0" style="display:block;margin:0 auto;"></iframe></div>');
		var readerAd3 = jQuery('<div id="ad3" class="adBox"><iframe src="/jsp_cn/ad/end_page.jsp?loc=3" width="760" height="90" scrolling="no" frameborder="0" style="display:block;margin:0 auto;"></iframe></div>');
		var readerAd4 = jQuery('<div id="ad4" class="adBox"><iframe src="/jsp_cn/ad/end_page.jsp?loc=4" width="760" height="90" scrolling="no" frameborder="0" style="display:block;margin:0 auto;"></iframe></div>');
		var readerAd5 = jQuery('<div id="ad5" class="adBox"><iframe src="/jsp_cn/ad/end_page.jsp?loc=5" width="760" height="90" scrolling="no" frameborder="0" style="display:block;margin:0 auto;"></iframe></div>');
			if(f == 1){
				//s1 s2
				if(jQuery("#page_1").length>0&&jQuery("#ad1").length==0){
					readerAd1.insertAfter(jQuery("#page_1"));				
				}
				if(jQuery("#page_2").length>0&&jQuery("#ad2").length==0){
					readerAd2.insertAfter(jQuery("#page_2"));
				}
				if(jQuery("#page_4").length>0&&jQuery("#ad3").length==0){
					readerAd3.insertAfter(jQuery("#page_4"));
				}
				if(jQuery("#page_6").length>0&&jQuery("#ad4").length==0){
					readerAd4.insertAfter(jQuery("#page_6"));
				}
				if(jQuery("#page_8").length>0&&jQuery("#ad5").length==0){
					readerAd5.insertAfter(jQuery("#page_8"));
				}
			}
			else if(f == 2){
				//s3 s4
				if(jQuery("#page_1").length>0&&jQuery("#ad1").length==0){
					readerAd1.appendTo(jQuery("#page_1"));				
				}
				if(jQuery("#page_2").length>0&&jQuery("#ad2").length==0){
					readerAd2.appendTo(jQuery("#page_2"));
				}
				if(jQuery("#page_4").length>0&&jQuery("#ad3").length==0){
					readerAd3.appendTo(jQuery("#page_4"));
				}
				if(jQuery("#page_6").length>0&&jQuery("#ad4").length==0){
					readerAd4.appendTo(jQuery("#page_6"));
				}
				if(jQuery("#page_8").length>0&&jQuery("#ad5").length==0){
					readerAd5.appendTo(jQuery("#page_8"));
				}
			}
	},
	setReaderLogin:function(){
		var _this = this;
		if(_this.config.isLogin == 1){return;}
		var l = jQuery("#contentcontainer").offset().left+jQuery("#contentcontainer").width()-200;
		jQuery(".reader_login").css({left:l}).show();
		var sc = jQuery(window).scrollTop()+jQuery(window).height(),
		sct = jQuery(document.body)[0].scrollHeight;
		if(_this.config.isBuy == 0){
			var oReaderLogin = jQuery(".reader_login");
			if(oReaderLogin.length>0){
				if(_this.config.showStyle == 2){
					jQuery(".reader_login").css('bottom',40);
					return false;	
				}
				if(sc>=sct){
					jQuery(".reader_login").css('bottom',200);
				}
				else{
					jQuery(".reader_login").css('bottom',40);
				}
			}
		}
	},
	setBeforeAd:function(){//前贴片广告显示
		var _this = this;
		var boforeAd = jQuery("#beforeAd");
		if(boforeAd.length==0){return;}
		var st = jQuery(window).scrollTop();
		if(st>_this.oSiderTop){
			boforeAd.css({position:'fixed'});
		}
		else{
			if(_this.zoomFlag == 0){
				boforeAd.css({position:'absolute'});
			}
			else if(_this.zoomFlag == 1){
				boforeAd.css({position:'fixed'});
			}
			
		}
		if(_this.zoomFlag == 1){
			boforeAd.css({width:1130});
		}
		else{
			boforeAd.css({width:802});
		}

		if(!window.XMLHttpRequest){
			boforeAd.css({position:'absolute'});
		}
	},
	countHide:function(num,numBox,o){//前贴片广告倒计时
		var boforeAd = jQuery("#beforeAd");
		if(boforeAd.length==0){return;}
		var _this = this;
		var contNum = num,
		numBox = jQuery("#"+numBox);
		numBox.html(contNum);
		//开始计数
		jQuery.post("/p/end/ajax/end_adv_log.do",{pid:_this.config.productId,flag:1,date:new Date().getTime(),page:_this.config.allPage,referer:_this.config.referer},function(data){});
		var numTimer = setInterval(function(){
			if(contNum == 1){
				clearInterval(numTimer);
				//消失之前
				jQuery.post("/p/end/ajax/end_adv_log.do",{pid:_this.config.productId,flag:2,date:new Date().getTime(),page:_this.config.allPage,referer:_this.config.referer},function(data){});
				jQuery("#"+o).remove();
				return;
			}
			contNum--;
			numBox.html(contNum);
		},1000);
	},
	handMove:function(){
		var oDiv=document.getElementById('contentcontainer');
		jQuery("#moveHandel").css({cursor:'url('+picture_image_path_v1+'/images_cn/news/html_reader/cursor_hand1.ico?rand=20140721),auto'});
		oDiv.onmousedown=function (ev){
		var oEvent=ev||event;	//处理事件对象兼容性问题
		var oEventY = oEvent.clientY;
		var oEventX = oEvent.clientX;
		var startY = oEventY+jQuery(window).scrollTop();
		var startX = oEventX+jQuery(window).scrollLeft();
		jQuery("#moveHandel").css({cursor:'url('+picture_image_path_v1+'/images_cn/news/html_reader/cursor_drag_hand1.ico?rand=20140721),auto'});
		if(oDiv.setCapture)
		{
			//IE
			oDiv.onmousemove=fnMove;
			oDiv.onmouseup=fnUp;	
			oDiv.setCapture();
		}
		else
		{
			//非IE
			oDiv.onmousemove=fnMove;
			oDiv.onmouseup=fnUp;
		}
		
		jQuery(".adBox").bind('mouseover',function(event){
			event.stopPropagation();
			fnUp();
		});
		// jQuery(".docin_reader_tools").bind('mouseover',function(event){
		// 	event.stopPropagation();
		// 	fnUp();
		// });
		
		jQuery(oDiv).bind('mouseleave',function(event){
			event.stopPropagation();
			fnUp();
		});
		function fnMove(ev)
		{	
			var sct = jQuery(document).height();
			var scw = jQuery(window).width()*1.5;
			var oEvent = ev||event;
			var st = jQuery(window).scrollTop();
			var sl = jQuery(window).scrollLeft();
			var t = (oEvent.clientY+st)-startY;
			var h = (oEvent.clientX+sl)-startX;
			var cH = jQuery(window).height();
			var cW = jQuery(window).width();
			var pct = cH/sct;
			var pcw = cW/scw;
			var ddd = (t/sct)*cH/pct;
			var eee = (h/scw)*cW/pcw;
			var moveT = Math.ceil(st-ddd);
			var moveL = Math.ceil(sl-eee);
			jQuery(window).scrollTop(moveT);
			jQuery(window).scrollLeft(moveL);
			return false;
		}
		function fnUp()
		{
			jQuery("#moveHandel").css({cursor:'url('+picture_image_path_v1+'/images_cn/news/html_reader/cursor_hand1.ico?rand=20140721),auto'});
			oDiv.onmousemove=null;
			oDiv.onmouseup=null;
			if(oDiv.releaseCapture)
			{
				oDiv.releaseCapture();
			}
			return false;
		}
		return false;
	};
	},
	hideSelected:function(){
		var _this = this;
		var aFlash = document.getElementById('contentcontainer').getElementsByTagName('object');
		jQuery(document).bind('click',function(){
			if(_this.config.showStyle == 2){return;}
			aFlash = document.getElementById('contentcontainer').getElementsByTagName('object');
			for(var i = 0;i<aFlash.length;i++){
				aFlash[i].hideSelected()
			}
		});
		jQuery(".panel").bind("click",function(ev){
			ev.stopPropagation(); //阻止冒泡
		});
	},
	tabChangeHand:function(){
		var _this = this;
		var oHandBtn = jQuery("#j_hand"),
		oSelectBtn = jQuery("#j_select");
		if(oHandBtn.length == 0){return;}
		oHandBtn.bind('click',function(){
			if(jQuery(this).hasClass('drag_hand_no')){return;}
			jQuery(this).addClass('btn_cur');
			oSelectBtn.removeClass('btn_cur');
			jQuery("#moveHandel").show();
			_this.handMove();
		});
		oSelectBtn.bind('click',function(){
			if(jQuery(this).hasClass('select_hand_no')){return;}
			jQuery(this).addClass('btn_cur');
			oHandBtn.removeClass('btn_cur');
			jQuery("#moveHandel").hide();
			var oDiv=document.getElementById('contentcontainer');
			oDiv.onmousedown = null;
			oDiv.onmousemove = null;
			oDiv.onmouseup = null;
		});
	},
	loadRelative:function(){
		var _this  = this;
		var disease = typeof(is_disease)=="undefined"?0:1;
		var url = "/app/p/end/ajax/getRightRelativeDocs";
		var metaKeyWords = jQuery("meta[name='keywords']").attr("content");
		var metaDescription = jQuery("meta[name='description']").attr("content");
		var isShowAdv = _this.config.showAd;
		jQuery.post(url,{is_disease:disease,type:1,metaDescription:metaDescription,metaKeyWords:metaKeyWords,id:_this.config.productId,isShowAdv:isShowAdv},function(re){
			//将请求结果放置到相关文档div中
				jQuery("#relative_doc_mod").append(re);
				// if(isShowAdv == false){
				// 	//vip用户
				// 	return false;
				// }
				if(jQuery(".sider_guanggao").length>0&&isShowAdv == true){
					jQuery(".sider_guanggao").show();
					var rightNewAd = jQuery('#right_end_new_adv'),
					oCloseCurMode = jQuery("#j_closeCurMod");
					if(rightNewAd.length>0){
						jQuery(window).bind("scroll",function(){
							var advTop = rightNewAd.offset().top,
								st = jQuery(window).scrollTop(),
								sH = st + jQuery(window).height();
								if(advTop<=sH){
									if(!jQuery('#right_end_new_adv iframe').attr('src')){
										jQuery('#right_end_new_adv iframe').attr({src:'/docin_adv/adv.do?pos=27'});
									}
								}
						});
					}
					if(oCloseCurMode.length>0&&jQuery(".sider_guanggao").length>0){
						oCloseCurMode.hover(function(){
							jQuery(".sider_guanggao").addClass("gg_hover");
						},function(){
							jQuery(".sider_guanggao").removeClass("gg_hover");
						});
						oCloseCurMode.bind("click",function(){
							if(_this.config.isLogin == 0){//未登录,放个cookie
								
								var cook = new CookieClass();
								cook.expires = 1;
								cook.domain = ".docin.com";
								cook.path = "/";
								cook.setCookie("vip_alert_adv", "1");
								
								changeCookieValue(cook,"vip_alert_adv");
								
//								showlogin();
								return false;
							}
							var openVipAlert = new CreateDocinDialog({
								id:"openVipTips",
								cls:"",
								title:"豆丁提示",
								content:'<p style="text-align:center;font-size:16px;font-weight:bold;height:32px;line-height:32px;margin-bottom:10px;">开通VIP，即享文档页零广告<p><p>开通VIP即享：阅读文档页零广告，还有可复制文档文本、下载免费文档不限量、专属妹子客服…等诸多<a style="text-decoration:underline;" href="http://www.docin.com/helpcenter/getProblem.do?pid=96" target="_blank" title="VIP特权">VIP特权</a><p>',
								cancel:{txt:"关闭",value:true},
								confirm:{txt:"开通VIP",value:true},
								callBack:{okBack:function(){openVipAlert = null;window.open('/app/jump_adv/pay_vip.do?channelid=0313&buyVipFrom=11');return true;},noBack:function(){openVipAlert = null;return false;}}
							});
						});
					}
				}
				relateDocH = jQuery('.relative_doc_inner').outerHeight();
				initRelaHeight();
				jQuery(window).bind('resize',function(){
					resizeRela();
				});
				if(!window.XMLHttpRequest){return;}			
		});
	},
	setFloderBtn:function(){
		var _this = this;
		var oLeft = jQuery(".main").offset().left+jQuery(".main").width();
		_this.floderBtn.css({left:oLeft});
	},
	setPage:function(window){
		var _this = this;
		var oSider = jQuery(".aside"),
		oToolBar = jQuery('.docin_reader_tools'),
		oSiderTop = _this.oSiderTop;
		oContWidth = jQuery("#contentcontainer").width(),
		oSiderRight = (jQuery(window).width()-1130)/2,
		oUserInfo = jQuery('.user_doc_mod');
		var oRelative = jQuery('.relative_doc_mod');
		var oFixControl = jQuery(".j_sidercontrol_build");//建筑新增代码

		if(oRelative.length>0){
			var oRelativeTop = oRelative.offset().top;
		}
		
		var oBanner = jQuery(".doc_hd_mini"),
			oBannerBuild = jQuery(".doc_hd");
		if(oBanner.length>0){
			var oBannerTop = oBanner.height();
		}
		if(oBannerBuild.length>0){
			var oBannerTop = oBannerBuild.height() + jQuery(".doc_hds").height();
		}

		setToolBarFix(1);
		siderFixedMove(1);
		_this.width = jQuery(window).width();
		_this.height = jQuery(window).height();
		jQuery(window).bind('scroll',function(){
			if((_this.zoomFlag == 0||_this.zoomFlag == 1)||_this.config.showStyle == 1){
				setBarPosition();
			}
			if(!window.XMLHttpRequest){return;}
			if(_this.zoomFlag == 1||_this.zoomFlag == 2){
				setToolBarFix(2);
				siderFixedMove(2);
			}
			else if(_this.zoomFlag == 0&&_this.config.showStyle == 1){
				setToolBarFix(1);
				siderFixedMove(1);
			}
			else if(_this.zoomFlag == 0&&_this.config.showStyle==2){
				setToolBarFix(1);
			}
			if(_this.zoomFlag == 0&&_this.config.showStyle == 1 && (typeof(isCartOrisCode)=="undefined"||typeof(isBuilding)=="undefined")){
				var st = jQuery(window).scrollTop();
				if(st>oSiderTop){
					resizeRelaHeight();				
				}
				else{
					initRelaHeight();
				}
			}
		});
		jQuery(window).bind('resize',function(){
			if(!window.XMLHttpRequest){return;}
			oSiderRight = (jQuery(window).width()-1130)/2;
			if(typeof(isBuilding)!="undefined"){
				_this.setFloderBtn();//建筑新增代码
			}
			if(jQuery(window).scrollTop()>oSiderTop){

				if(_this.zoomFlag == 0&&_this.config.showStyle == 1){//初始化状态
					siderFixedMove(1);
					return false;
				}
				if(_this.zoomFlag == 1||_this.zoomFlag == 2){
					siderFixedMove(2);
					return false;
				}
			}
			else{
				if(_this.zoomFlag == 0){//初始化状态
					siderFixedMove(3);
					return false;
				}
				if(_this.zoomFlag == 1||_this.zoomFlag == 2){
					siderFixedMove(2);
					return false;
				}
			}
			
		});
		function siderFixedMove(f){
			if(!window.XMLHttpRequest){return;}
			var st = jQuery(window).scrollTop();
			oSiderRight = (jQuery(window).width()-1130)/2;


			//建筑终极页新增代码开始
			if(typeof(isBuilding)!="undefined"&&isBuilding ==1){
				var cH = jQuery(window).height(),
				posb = oFixControl.offset().top+oFixControl.height(),
				sth = cH + st,
				post = oFixControl.offset().top;
				if(f == 1){
					if(st >oSiderTop){
						if(oSiderRight<=0){
							oSider.css({margin:0});
						}
						else{
							oSider.css({margin:0});
						}
					}
				}
				else if(f == 2){
					if(oSiderRight<=0){
						oSider.css({margin:0});
					}
					else{
						oSider.css({margin:0});
						oSiderRight = (jQuery(window).width()-1130)/2;
					}
				}
				oUserInfo.show();
				if(_this.config.showStyle == 1&&_this.zoomFlag == 0){
					relativeControl();
				}
				return false;
			}
			//建筑终极页新增代码结束


			if(f == 1){ 
				if(st >oSiderTop){
					if(oSiderRight<=0){
						oSider.css({position:'fixed',top:56,right:oSiderRight*2,left:'auto',margin:0});
					}
					else{
						oSider.css({position:'fixed',top:56,right:oSiderRight,left:'auto',margin:0});
					}
					if(typeof(isCartOrisCode)!="undefined"&&isCartOrisCode ==1){
						oUserInfo.show();
					}
					else{
						oUserInfo.hide();
					}
				}
				else{
					oSider.css({position:'relative',top:0,right:0,left:'auto',marginTop:8});
					oUserInfo.show();
				}
			}
			else if(f == 2){
				oUserInfo.hide();
				if(oSiderRight<=0){
					oSider.css({position:'fixed',top:56,right:oSiderRight*2,left:'auto',margin:0});
				}
				else{
					oSider.css({position:'fixed',top:56,margin:0,left:'auto'});
					oSiderRight = (jQuery(window).width()-1130)/2;
					oSider.css({right:oSiderRight});
				}
				return;
				
			}
			else if(f == 3){
				oSider.css({position:'relative',top:0,right:0,left:'auto',marginTop:8});
				return;
			}
		}
		function relativeControl(){
			if(!window.XMLHttpRequest){return;}
			var st = jQuery(window).scrollTop();
			cH = jQuery(window).height(),
			posb = oFixControl.offset().top+oFixControl.height(),
			sth = cH + st,
			post = oFixControl.offset().top;
			var oFixTop = oFixControl.offset().top;
			if(st>=_this.oSiderTop){
				if(oFixControl.height()<=cH){
					oFixControl.css({position:'fixed',top:50,bottom:'auto'});
					return;
				}
				if(st>sst){//向下
					if(ft>=0){
						//1、先判断右侧内容看到最后，即让他居底部定位
						posb = oFixControl.offset().top+oFixControl.height();
						if(posb<=sth){
							oFixControl.css({position:'fixed',bottom:0,top:'auto'});
							oFixTop = oFixControl.offset().top;
							ftup = oFixTop - 140;
						}
						else{
							oFixControl.css({position:'relative',bottom:'auto',top:ft});
						}
					}
				}
				else{//向上
					if(ftup>=0){
						if(oFixTop>=st){
							oFixControl.css({position:'fixed',top:50,bottom:'auto'});
							oFixTop = oFixControl.offset().top;
							ft = oFixTop - 140;
						}
						else{
							oFixControl.css({position:'relative',top:ftup,bottom:'auto'});
						}
						
					}
				}

			}
			else{
				oFixControl.css({position:'relative',top:0,bottom:'auto'});
				ft = ftup = 0;
			}
			sst = st;
		}
		function setToolBarFix(f){
			if(!window.XMLHttpRequest){return;}
			var st = jQuery(window).scrollTop();
			if(f == 1){
				if(st >= oBannerTop){
					oToolBar.css({top:0});
					if(_this.zoomFlag == 0){
						jQuery("#bookmarktips").css({top:41,width:800});
					}
					else{
						jQuery("#bookmarktips").css({top:41,width:'auto'});
					}
					
					return;
				}
				else{
					var tt1 = jQuery("#contentcontainer").offset().top-60;
					jQuery("#bookmarktips").css({top:tt1,width:800});
					oToolBar.css({top:oBannerTop-st});
				}
			}
			else if(f == 2){
				oToolBar.css({top:0});
				jQuery("#bookmarktips").css({top:41,width:'auto'});
					return;				
			}
		}
		window.setToolBarFix = setToolBarFix;
		window.siderFixedMove = siderFixedMove;
		_this.relativeControl = relativeControl;
	}
}
var docinReader = new DocinReaderPlayer(readerConfig);
(function($){
	$.fn.scrollLoading = function(options) {
		var defaults = {
			attr: "data-url"	
		};
		var params = $.extend({}, defaults, options || {});
		params.cache = [];
		$(this).each(function() {
			var node = this.nodeName.toLowerCase(), url = $(this).attr(params["attr"]);
			//if (!url) { return; }
			//重组
			var data = {
				obj: $(this),
				tag: node,
				url: url
			};
			params.cache.push(data);
		});
		
		//动态显示数据
		var loading = function() {
			var st = $(window).scrollTop(), sth = st + $(window).height();
			var ddd = [];//记录页码
			$.each(params.cache, function(i, data) {
				var o = data.obj, tag = data.tag, url = data.url;
				if (o) {

						post = o.offset().top; posb = post + o.height();

						var a = (post > st) && ((post-st)<$(window).height()/2);
						var b = (posb < sth)&&((posb-st)>$(window).height()/2);
						var c = (post<sth)&&(posb>sth)&&(docinReader.curPage == 1);
						if(a||b||c){
							var t = i+1;
							if (tag === "img") {
								//图片，改变src
								o.attr("src", url);	
							}else {
									// o.load(url);	
								if(log_preNum != t){
									docinReader.addFlash(t);
									docinReader.clearFlash(t);
									jQuery('.page_cur').val(t);
									docinReader.curPage = t;
									docinReader.pageButton(t);
									docinReader.bookMarkShow(t);
									log_preNum = t;
									if(docinReader.config.isBuy == 1){
										if(typeof fromWebSearch !="undefined"&&docinReader.curPage == docinReader.config.allPage){

											fromWebSearch();
										}
									}
									else if(docinReader.config.isBuy == 0){
										if(typeof fromWebSearch !="undefined"&&docinReader.curPage == docinReader.config.previewNum){
											fromWebSearch();
										}
									}
									// var sc = jQuery(window).scrollTop()+jQuery(window).height(),
									// sct = jQuery(document.body)[0].scrollHeight;
									// if(sc>=sct){
									// 	if(docinReader.config.isBuy == 0){
									// 		jQuery('.page_cur').val(docinReader.config.previewNum);
									// 		docinReader.curPage = docinReader.config.previewNum;
									// 		docinReader.pageButton(docinReader.config.previewNum);
									// 		docinReader.bookMarkShow(docinReader.config.previewNum);
									// 		return;
									// 	}
									// 	else if(docinReader.config.isBuy == 1){
									// 		docinReader.curPage = docinReader.config.allPage;
									// 		jQuery('.page_cur').val(docinReader.config.allPage);
									// 		docinReader.pageButton(docinReader.config.allPage);
									// 		docinReader.bookMarkShow(docinReader.config.allPage);
									// 		return;
									// 	}
									// }
								}

							}
						}
				}
			});	
			scrollFlag = true;
			return false;	
		};
		//事件触发
		//加载完毕即执行
		if(docinReader.curPage == 1||docinReader.curPage ==readerConfig.previewNum){
			loading();
		}
		//loading();
		//滚动执行
		$(window).bind("scroll",function(){
			if(docinReader.config.showStyle == 1){
				if(scrollFlag == true){
					scrollFlag = false;
					loading();
				}
				
				//setTimeout(function(){loading();},100);
			}

			// if(docinReader.config.showStyle == 1&&scrollFlag == true){
			// 	setTimeout(function(){loading();},100);
			// 	scrollFlag = false;
			// }
		});

	};
})(jQuery);
(function(){
	var oBtnShowAll = jQuery("#openAll"),
	oMoreDec = jQuery("#moreDec"),
	oElipDec = jQuery("#elipDec"),
	oInfoArea = jQuery(".user_doc_bd"),
	oModelArea = jQuery(".user_doc_mod");
	if(oBtnShowAll.length>0){
		oBtnShowAll.bind('click',function(){
			if(jQuery(".user_doc_mod .inner").hasClass('innerShow')){
				closeMoreHandel();
			}
			else{
				jQuery(".user_doc_mod .inner").addClass('innerShow');
				if(oMoreDec.length>0){
					oElipDec.hide();
					oMoreDec.show();
				}
				oBtnShowAll.html('收起');
			}
			//return false;
		});
		// oModelArea.bind('mouseleave',function(event){
		// 	closeMoreHandel();
		// });
		function closeMoreHandel(){
			jQuery(".user_doc_mod .inner").removeClass('innerShow');
			if(oMoreDec.length>0){
				oMoreDec.hide();
				oElipDec.show();
			}
			oBtnShowAll.html('全部');
		}
	}
})();
function resizeRela(){
	var cH = jQuery(window).height();//屏幕高度
	if(cH < 310){
		return false;
	}
	if(jQuery(".sider_guanggao").length > 0){
		var adHeight = jQuery(".sider_guanggao").height();
	}
	else{
		var adHeight = 0;//vip
	}
	if(jQuery(window).scrollTop()>docinReader.oSiderTop){
		//var activeHeight = cH - 56 - 145 -10;//相关文档高度
		// var activeHeight = cH - adHeight -56;
		// if(relateDocH > activeHeight){
		// 	jQuery(".relative_doc_inner").css({height:activeHeight-2});
		// }
		// else{
		// 	jQuery(".relative_doc_inner").css({height:'auto'});
		// }
		resizeRelaHeight();
	}
	else{
		initRelaHeight();
	}
}

function initRelaHeight(){
	var cH = jQuery(window).height();
	if(cH < 310){
		return false;
	}
	var initActiveHeight = cH - oRelaTop;
	if(jQuery(".relative_bd .list_item").length>0){
		var oItemHeight = jQuery(".relative_bd .list_item").eq(1).outerHeight();
		var showNum = Math.floor((initActiveHeight-40-jQuery(".relative_bd .list_item .item_show").outerHeight())/oItemHeight);
		initActiveHeight = showNum*oItemHeight+40+jQuery(".relative_bd .list_item .item_show").outerHeight();
	}
	if(relateDocH > initActiveHeight){
		if(relateDocH < (jQuery(".relative_bd .list_item").eq(1).outerHeight()*2 +jQuery(".relative_bd .list_item .item_show").outerHeight())){
			var iH = jQuery(".relative_bd .list_item").eq(1).outerHeight()*2 +jQuery(".relative_bd .list_item .item_show").outerHeight();
			jQuery(".relative_doc_inner").css({height:iH});
		}else{
			jQuery(".relative_doc_inner").css({height:initActiveHeight-2});
		}
	}
	else{
		jQuery(".relative_doc_inner").css({height:'auto'});
	}
}
function resizeRelaHeight(){
	var cH = jQuery(window).height();
	if(cH < 310){
		return;
	}
	if(jQuery(".sider_guanggao").length > 0){
		var adHeight = jQuery(".sider_guanggao").height();
	}
	else{
		var adHeight = 0;//vip
	}
	var activeHeight = cH - adHeight -56;
	if(jQuery(".relative_bd .list_item").length>0){
		var oItemHeight = jQuery(".relative_bd .list_item").eq(1).outerHeight();
		var showNum = Math.floor((activeHeight-40-jQuery(".relative_bd .list_item .item_show").outerHeight())/oItemHeight);
		activeHeight = showNum*oItemHeight+40+jQuery(".relative_bd .list_item .item_show").outerHeight();
	}
	if(relateDocH > activeHeight){
		if(relateDocH < (jQuery(".relative_bd .list_item").eq(1).outerHeight()*2 +jQuery(".relative_bd .list_item .item_show").outerHeight())){
			var iH = jQuery(".relative_bd .list_item").eq(1).outerHeight()*2 +jQuery(".relative_bd .list_item .item_show").outerHeight();
			jQuery(".relative_doc_inner").css({height:iH});
		}else{
			jQuery(".relative_doc_inner").css({height:activeHeight-2});
		}
	}
	else{
		jQuery(".relative_doc_inner").css({height:'auto'});
	}
}

function hideOtherSelected(){
	var aFlash = document.getElementById('contentcontainer').getElementsByTagName('object');
	for(var i = 0;i<aFlash.length;i++){
		aFlash[i].hideSelected();
	}
}

function likeTooModControl(f){
	if(f == 1){
		if(jQuery('#jControlDiv').length>0){
			jQuery('#jControlDiv').show();
		}
		else if(jQuery("#likeToo").length>0){
			jQuery("#likeToo").show();
		}
	}
	else if(f == 2){
		if(jQuery('#jControlDiv').length>0){
			jQuery('#jControlDiv').hide();
		}
		else if(jQuery("#likeToo").length>0){
			jQuery("#likeToo").hide();
		}
	}
}
function controlLeftBottomAdv(f){
	if(jQuery(".left_buttom_adv").length == 0){return;}
	if(f == 1){
		jQuery(".left_buttom_adv").show();
	}
	else if(f == 2){
		jQuery(".left_buttom_adv").hide();
	}
}
//是否有权限复制
function isAuthForCopy(){
		if(typeof readerConfig.isBD == "undefined"){return 0;}
		if(readerConfig.isBD == 0){
			if(readerConfig.isLogin == 0){//未登录
				//放个cookie
				var cook = new CookieClass();
				cook.expires = 1;
				cook.domain = ".docin.com";
				cook.path = "/";
				cook.setCookie("can_copy_alert", "1");
				changeCookieValue(cook,"can_copy_alert");
				
				return 0;
			}
			else if(readerConfig.showAd == true){
				var openVipAlert = new CreateDocinDialog({
					id:"openVipTips",
					cls:"",
					title:"豆丁提示",
					content:'<p style="text-align:center;font-size:16px;font-weight:bold;height:32px;line-height:32px;margin-bottom:10px;">您还不是VIP，不能使用复制功能<p><p>开通VIP即享：可复制文档文本，还有阅读文档页零广告、下载免费文档不限量、专属妹子客服…等诸多<a style="text-decoration:underline;" href="http://www.docin.com/helpcenter/getProblem.do?pid=96" target="_blank" title="VIP特权">VIP特权</a><p>',
					cancel:{txt:"关闭",value:true},
					confirm:{txt:"开通VIP",value:true},
					callBack:{okBack:function(){openVipAlert = null;window.open('http://www.docin.com/app/jump_adv/pay_vip.do?channelid=0313&buyVipFrom=12');return true;},noBack:function(){openVipAlert = null;return false;}}
				});
				
				
				
				return 0;
			}
			else if(readerConfig.showAd == false){
				return 1;
			}
		}
		return 1;
	}



	function checkCookie_vip(){
		var cook = new CookieClass();
		var remindClickId = cook.getCookie("vip_alert_adv");
		if(remindClickId == "1" && tmp_userId > 0 && isAdvShow == "true"){//不是vip，显示弹层
			
			var openVipAlert = new CreateDocinDialog({
				id:"openVipTips",
				cls:"",
				title:"豆丁提示",
				content:'<p style="text-align:center;font-size:16px;font-weight:bold;height:32px;line-height:32px;margin-bottom:10px;">开通VIP，即享文档页零广告<p><p>开通VIP即享：阅读文档页零广告，还有可复制文档文本、下载免费文档不限量、专属妹子客服…等诸多<a style="text-decoration:underline;" href="http://www.docin.com/app/jump_adv/pay_vip.do?channelid=0313&buyVipFrom=11" target="_blank" title="VIP特权">VIP特权</a><p>',
				cancel:{txt:"关闭",value:true},
				confirm:{txt:"开通VIP",value:true},
				callBack:{okBack:function(){openVipAlert = null;window.open('/app/jump_adv/pay_vip.do?channelid=0313&buyVipFrom=11');return true;},noBack:function(){openVipAlert = null;return false;}}
			});
		}
		
		cook.expires = 1;
		cook.domain = ".docin.com";
		cook.path = "/";
		cook.setCookie("vip_alert_adv", -1);
	}
	function checkCookie_copy(){
		var cook = new CookieClass();
		var remindClickId = cook.getCookie("can_copy_alert");
		if(remindClickId == "1" && tmp_userId > 0 && isAdvShow == "true"){//不是vip，显示弹层
			
			var openVipAlert = new CreateDocinDialog({
				id:"openVipTips",
				cls:"",
				title:"豆丁提示",
				content:'<p style="text-align:center;font-size:16px;font-weight:bold;height:32px;line-height:32px;margin-bottom:10px;">您还不是VIP，不能使用复制功能<p><p>开通VIP即享：可复制文档文本，还有阅读文档页零广告、下载免费文档不限量、专属妹子客服…等诸多<a style="text-decoration:underline;" href="http://www.docin.com/helpcenter/getProblem.do?pid=96" target="_blank" title="VIP特权">VIP特权</a><p>',
				cancel:{txt:"关闭",value:true},
				confirm:{txt:"开通VIP",value:true},
				callBack:{okBack:function(){openVipAlert = null;window.open('http://www.docin.com/app/jump_adv/pay_vip.do?channelid=0313&buyVipFrom=12');return true;},noBack:function(){openVipAlert = null;return false;}}
			});
		}
		cook.expires = 1;
		cook.domain = ".docin.com";
		cook.path = "/";
		cook.setCookie("can_copy_alert", -1);
	}
	checkCookie_vip();
	checkCookie_copy();

function flashChecker(){
	var hasFlash=0;　　　　//是否安装了flash
	var flashVersion=0;　　//flash版本

	if(document.all){
		var swf = new ActiveXObject('ShockwaveFlash.ShockwaveFlash');
		if(swf) {
			hasFlash=1;
			VSwf=swf.GetVariable("$version");
			flashVersion=parseInt(VSwf.split(" ")[1].split(",")[0]);
		}
	}else{
		if (navigator.plugins && navigator.plugins.length > 0){
			var swf=navigator.plugins["Shockwave Flash"];
			if (swf){

				hasFlash=1;
				var words = swf.description.split(" ");

				for (var i = 0; i < words.length; ++i){
			         if (isNaN(parseInt(words[i]))) continue;
			         flashVersion = parseInt(words[i]);
				}
			}
		}
	}
	return {f:hasFlash,v:flashVersion};
}

(function(){
	var fls=flashChecker();
	if(fls.f){
		//console.log("您安装了flash,当前flash版本为: "+fls.v+".x")
	}else{
		jQuery('<div id="unFlashTips" style="position:absolute;left:0;top:0;z-index:2;width:100%;height:200px;padding-top:160px;text-align:center;background:url('+picture_image_path_v1+'/images_cn/error/error-flash-tips.jpg) no-repeat 50% 70px">您的计算机尚未安装Flash，<a href="https://get.adobe.com/cn/flashplayer/" target="_blank">点击安装https://get.adobe.com/cn/flashplayer/</a></div>').appendTo('.doc_reader_mod');
	}
}())