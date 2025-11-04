import io
import json
import math
from pathlib import Path

import pylab
import scipy.stats as stats

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# -------------------- OUTPUT KLASÖRÜ --------------------
OUT = Path("Linear_Regression_Outputs")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- VERİ --------------------
df = pd.read_csv('data/EcommerceCustomers.csv')

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------- MODEL --------------------
lm = LinearRegression()
lm.fit(x_train, y_train)

# Katsayılar
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
print("\nCoefficients:\n", cdf)
cdf.to_csv(OUT / "coefficients.csv")  # katsayıları kaydet

# Tahmin ve artıklar
predictions = lm.predict(x_test)
residuals = y_test - predictions

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": predictions,
    "residual": residuals.values
})
pred_df.to_csv(OUT / "predictions_residuals.csv", index=False)

# -------------------- METRİKLER --------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)

metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse}
print(metrics)

# JSON ve TXT olarak yaz
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(OUT / "metrics.txt").write_text(
    f"Mean Absolute Error: {mae}\nMean Squared Error: {mse}\nRMSE: {rmse}\n",
    encoding="utf-8"
)

# -------------------- GÖRSELLEŞTİRME KAYITLARI --------------------
# 1) Artık dağılımı
g = sns.displot(residuals, bins=30, kde=True)
plt.title("Residuals Distribution")
g.figure.savefig(OUT / "residuals_distribution.png", dpi=150, bbox_inches="tight")
plt.close('all')

# 2) QQ-Plot
fig = plt.figure()
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals")
fig.savefig(OUT / "qq_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3) Keşif amaçlı: head/info/describe dosyaları
df.head().to_csv(OUT / "df_head.csv", index=False)
buf = io.StringIO()
df.info(buf=buf)
(OUT / "df_info.txt").write_text(buf.getvalue(), encoding="utf-8")
df.describe().to_csv(OUT / "df_describe.csv")

# 4) Jointplot'lar
g = sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)
g.figure.savefig(OUT / "joint_time_on_website_vs_spent.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)

g = sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)
g.figure.savefig(OUT / "joint_time_on_app_vs_spent.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)

# 5) Pairplot
g = sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})
g.figure.savefig(OUT / "pairplot.png", dpi=120, bbox_inches="tight")
plt.close('all')

# 6) lmplot: Length of Membership vs Spent
g = sns.lmplot(
    x='Length of Membership',
    y='Yearly Amount Spent',
    data=df,
    scatter_kws={'alpha': 0.3}
)
g.figure.savefig(OUT / "lmplot_length_membership_vs_spent.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)

# 7) Tahmin vs Gerçek
plt.figure()
sns.scatterplot(x=predictions, y=y_test)
plt.xlabel("Predictions")
plt.ylabel("Actual")
plt.title("Evaluation of Linear Regression Model")
plt.tight_layout()
plt.savefig(OUT / "pred_vs_actual.png", dpi=150)
plt.close()

print(f"Tüm çıktılar '{OUT.resolve()}' klasörüne kaydedildi.")
