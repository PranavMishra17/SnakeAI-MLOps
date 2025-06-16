:: Train individual algorithms
echo 🚀 Starting training of individual algorithms..
echo ============================================
echo Training DQN...

python src/dqn_trainer.py

echo ============================================

echo Training Policy Gradient...
python src/policy_gradient_trainer.py

echo ============================================

echo Training Actor-Critic...
python src/actor_critic_trainer.py

:: Or use the orchestrator
::python src/train_models.py --technique all


echo ============================================
echo Training completed successfully!
echo ============================================
echo
echo 📊 Starting evaluation of trained models...
:: Enhanced evaluation
python src/model_evaluator.py