# Phase 1 — Reliable Model Selection & Backtesting

این فاز **روش‌های موجود را حذف نمی‌کند**؛ تمام Candidateهای آماری، CatBoost، Price-Volume و Blend همچنان در شبکه رقابت هستند. هدف آن است که بهبود تصادفی در یک مبدأ تاریخی، به اشتباه به‌عنوان برتری پایدار پذیرفته نشود.

## معماری

- \`selection_core.py\`: مشخصات Candidateها (ID یکتا، خانواده، نام، شاخص‌های پشتیبانی‌شده، دوره، حداقل تاریخچه، هزینه محاسباتی و نوع Cache)، برنامه Rolling-Origin و قواعد انتخاب Champion.
- \`auto_engine.py\`: ایجاد Predictionهای تاریخی در همان Global Model، رتبه‌بندی Candidateها روی هدف‌های یکسان، انتخاب Guarded Champion، و نگه‌داشتن Legacy Global به‌عنوان Benchmark.
- \`category_selection.py\`: انتخاب Recipe قابل‌اتکا در Subcategoryهای ۱ تا ۳، با Fallback به والد یا Global؛ نتایج مستقل Holdout از انتخاب Recipe حذف می‌مانند.
- \`benchmark_levels.py\`: سنجش سطح Market، شاخه‌های Subcategory، Brand، تک‌سری و Demand Pattern. هر سطح گزارش مجزاست؛ مجموع سطوح هم‌پوشان نباید جمع شود.
- \`evaluate_hierarchy.py\`: گزارش CSV فشرده و CSV تفکیکی برای مقایسه مستقل.
- \`app_ui.html\`: نمایش منطق Champion، تفاوت WAPE انتخاب با WAPE Holdout و مقایسه Legacy/Guarded/Hierarchical.

## قواعد انتخاب Global Champion

1. هر Candidate روی مبدأها و هدف‌های یکسان ارزیابی می‌شود؛ WAPE از **مجموع خطای مطلق سلول‌های مشترک** تقسیم بر مجموع Actual همان سلول‌ها محاسبه می‌شود.
2. بهترین Candidate آماریِ غیر ML، **Reference** محافظه‌کار است.
3. Challenger فقط زمانی Reference را کنار می‌زند که:
   - مجموع خطای مطلق آن دست‌کم **۲٪** نسبت به Reference کمتر باشد؛
   - در حداقل ۲ مبدأ و حداقل دو‌سوم مبدأها خطای پایین‌تری داشته باشد؛
   - در هیچ مبدأ، بدترشدن نسبی خطا بیش از **۵٪** نسبت به Reference نباشد.
4. در صورت کافی‌نبودن شواهد، Reference باقی می‌ماند؛ در انتخاب Category نیز قواعد محافظ موجود در \`CategoryGuard\` حفظ شده‌اند.
5. این آستانه‌ها قواعد عملیاتی پیش‌فرض هستند و از Holdout تنظیم نمی‌شوند؛ بهینه‌بودن آن‌ها روی داده اختصاصی EMRC هنوز اثبات نشده است.

**نکته:** این روش «کمترین WAPE تحت شروط پایداری» را انتخاب می‌کند، نه صرفاً کمترین WAPE بدون قید. ممکن است Recipe منتخب در جدول رتبه‌بندی اولین نباشد و این مسئله در UI توضیح داده می‌شود.

## برنامه زمانی

برای Dataset دارای Category معتبر:

- اگر حداقل ۳۰ دوره کامل وجود داشته باشد: مبدأ Holdout = سه دوره قبل از آخرین دوره و مبدأهای انتخاب در فواصل سه‌دوره‌ای قبل از آن قرار می‌گیرند.
- حداکثر ۳ مبدأ Selection؛ در تاریخچه ۳۶ دوره‌ای مبدأهای \`cutoff-12\`، \`cutoff-9\`، \`cutoff-6\` و Holdout برابر \`cutoff-3\` است.
- اهداف H1، H2 و H3 برای همه Candidateها تکمیل‌شده و در هر مقایسه یکسان هستند.
- \`monthly\` و \`bimonthly\` بر اساس **شماره مشاهده** حرکت می‌کنند؛ دوره دوماهه به‌معنای هر ۲ ماه است، نه هر ماه.
- تاریخچه کوتاه، نبود Category یا گزینه \`Global-only\`: همان انتخاب Global قبلی با دو مبدأ، **بدون ادعای Holdout مستقل**.
- Holdout فقط **برای گزارش** است؛ هیچ معیار/آستانه‌ای با آن تنظیم نمی‌شود.

## تعریف Baseline منصفانه

سه خروجی روی همان Holdout گزارش می‌شوند:

1. \`legacy_global\`: انتخاب مدل قبلی با **کمترین WAPE تجمیعی روی همان مبدأهای Selection**.
2. \`global\`: Global Champion محافظه‌کار با معیار پایداری.
3. \`hierarchical\`: Global Champion محافظه‌کار به‌اضافه Recipeهای اختصاصی Category واجد شرایط.

Baseline اول از نظر **قاعده انتخاب** همان Global قبلی است، اما برای مقایسه منصفانه از مبدأهای Selection مشترک استفاده می‌کند. به این دلیل ممکن است با خروجی Global-only قدیمی که از مبدأهای تاریخی متفاوت استفاده کرده، یکسان نباشد. هیچ ادعایی درباره بهبود روی داده واقعی تا قبل از اجرای Benchmark وجود ندارد.

## معیارها و دامنه تحلیل

- WAPE، MAE، Bias، Actual، Forecast، خطای مطلق، نسبت Forecast/Actual (در صورت تعریف‌پذیربودن).
- در گزارش گروهی: Series-level WAPE در برابر Aggregated-group WAPE.
- صفر بودن مجموع Actual: WAPE/Bias برابر \`N/A\` است؛ MAE و Absolute Error همچنان گزارش می‌شوند.
- تعداد گروه‌های بهتر/بدتر و صفرهای مشاهده‌شده.
- طبقه‌بندی Sparse/Zero/Regular صرفاً با تاریخچه قبل از Holdout Origin، نه آینده.
- سطح \`Series\` یک ردیف واقعی مکان × محصول است؛ نباید بدون بررسی \`Element Caption\` به‌عنوان SKU یکتا تفسیر شود.

## اجرای تست و گزارش روی دستگاه خودت

در ریشه Repository، پس از نصب \`forecasting/requirements.txt\`:

\`\`\`powershell
python -m compileall -q forecasting
python -m unittest discover -s forecasting -p "test_*.py" -v
python forecasting/evaluate_hierarchy.py "C:\Data\master.xlsx" --periodicity monthly --market All --metric volume_kg --output outputs\phase1_volume.csv
python forecasting/evaluate_hierarchy.py "C:\Data\master.xlsx" --periodicity monthly --market All --metric value_1000_toman --output outputs\phase1_value.csv
\`\`\`

برای داده‌های دوماهه از \`--periodicity bimonthly\` استفاده کن و اگر Workbook دارای \`Element Caption\` هم‌پوشان است، \`--grain SKU\` یا سطح واقعی را مشخص کن.

خروجی دو فایل در هر اجراست:

- \`phase1_volume.csv\`: WAPE/MAE/Bias در سه افق و سه Strategy.
- \`phase1_volume_by_group.csv\`: گزارش تفکیکی Market، Category، Brand، Series و Demand Pattern؛ اعداد زیربخش‌های سلسله‌مراتبی **جمع‌پذیر بین سطوح مختلف نیستند**.

## معیار پذیرش و محدودیت‌ها

- چرخه کامل پیش‌بینی UI و Full Excel باید پس از Pull با داده مصنوعی و Workbook واقعی تست شود.
- تست‌های مستقل داخل Repository برای انتخاب محافظه‌کار، Rolling-Origin، تشخیص Leakage، Hierarchical Fallback و سازگاری خروجی باقی‌اند.
- طول تاریخچه محدود ممکن است اجازه مقایسه مستقل ندهد؛ یک Holdout منفرد نیز تضمین عملکرد آینده نیست.
- مرحله اول فقط روش انتخاب و اعتبارسنجی را بهبود می‌دهد؛ Intermittent Demand، Change Detection، ویژگی‌های Distribution و Forecast Intervals مربوط به فازهای بعدی‌اند.
- خروجی CSV و Master محرمانه باید فقط محلی باقی بمانند.
