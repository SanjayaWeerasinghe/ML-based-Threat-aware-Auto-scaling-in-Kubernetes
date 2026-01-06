# ML Model Training Scripts

## Version Structure

### v1/ - Baseline Models (5 Models)
1. `train_isolation_forest.py` - Tree-based anomaly detection
2. `train_one_class_svm.py` - Boundary-based anomaly detection
3. `train_knn.py` - Distance-based anomaly detection (🏆 Winner)
4. `train_gmm.py` - Probabilistic anomaly detection
5. `train_autoencoder.py` - Neural network-based anomaly detection

**Results:** KNN achieved best performance (78.53% accuracy, 99.13% precision, 2.24% FPR)

**Status:** ✅ Complete - All models trained and evaluated

### v2/ - Enhanced Models (Coming Soon)
Planned improvements:
- Hyperparameter tuning
- Ensemble methods
- Feature selection optimization
- Additional model architectures
- Production-ready inference pipelines

---

**Current Version:** v1
**Next Version:** v2 (In Development)
**Best Model:** KNN (Local Outlier Factor)
