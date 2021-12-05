/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define DAC_BUFF_LEN	1024
#define  DMA_BUFF_LEN	16
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
DAC_HandleTypeDef hdac;
DMA_HandleTypeDef hdma_dac2;

TIM_HandleTypeDef htim2;

/* USER CODE BEGIN PV */
uint16_t dma_buff[DMA_BUFF_LEN];
uint16_t dac_data[DAC_BUFF_LEN] = {2048,2060,2073,2085,2098,2110,2123,2135,2148,2161,2173,2186,2198,2211,2223,2236,
		2248,2261,2273,2286,2298,2311,2323,2335,2348,2360,2373,2385,2398,2410,2422,2435,
		2447,2459,2472,2484,2496,2508,2521,2533,2545,2557,2569,2581,2594,2606,2618,2630,
		2642,2654,2666,2678,2690,2702,2714,2725,2737,2749,2761,2773,2784,2796,2808,2819,
		2831,2843,2854,2866,2877,2889,2900,2912,2923,2934,2946,2957,2968,2979,2990,3002,
		3013,3024,3035,3046,3057,3068,3078,3089,3100,3111,3122,3132,3143,3154,3164,3175,
		3185,3195,3206,3216,3226,3237,3247,3257,3267,3277,3287,3297,3307,3317,3327,3337,
		3346,3356,3366,3375,3385,3394,3404,3413,3423,3432,3441,3450,3459,3468,3477,3486,
		3495,3504,3513,3522,3530,3539,3548,3556,3565,3573,3581,3590,3598,3606,3614,3622,
		3630,3638,3646,3654,3662,3669,3677,3685,3692,3700,3707,3714,3722,3729,3736,3743,
		3750,3757,3764,3771,3777,3784,3791,3797,3804,3810,3816,3823,3829,3835,3841,3847,
		3853,3859,3865,3871,3876,3882,3888,3893,3898,3904,3909,3914,3919,3924,3929,3934,
		3939,3944,3949,3953,3958,3962,3967,3971,3975,3980,3984,3988,3992,3996,3999,4003,
		4007,4010,4014,4017,4021,4024,4027,4031,4034,4037,4040,4042,4045,4048,4051,4053,
		4056,4058,4060,4063,4065,4067,4069,4071,4073,4075,4076,4078,4080,4081,4083,4084,
		4085,4086,4087,4088,4089,4090,4091,4092,4093,4093,4094,4094,4094,4095,4095,4095,
		4095,4095,4095,4095,4094,4094,4094,4093,4093,4092,4091,4090,4089,4088,4087,4086,
		4085,4084,4083,4081,4080,4078,4076,4075,4073,4071,4069,4067,4065,4063,4060,4058,
		4056,4053,4051,4048,4045,4042,4040,4037,4034,4031,4027,4024,4021,4017,4014,4010,
		4007,4003,3999,3996,3992,3988,3984,3980,3975,3971,3967,3962,3958,3953,3949,3944,
		3939,3934,3929,3924,3919,3914,3909,3904,3898,3893,3888,3882,3876,3871,3865,3859,
		3853,3847,3841,3835,3829,3823,3816,3810,3804,3797,3791,3784,3777,3771,3764,3757,
		3750,3743,3736,3729,3722,3714,3707,3700,3692,3685,3677,3669,3662,3654,3646,3638,
		3630,3622,3614,3606,3598,3590,3581,3573,3565,3556,3548,3539,3530,3522,3513,3504,
		3495,3486,3477,3468,3459,3450,3441,3432,3423,3413,3404,3394,3385,3375,3366,3356,
		3346,3337,3327,3317,3307,3297,3287,3277,3267,3257,3247,3237,3226,3216,3206,3195,
		3185,3175,3164,3154,3143,3132,3122,3111,3100,3089,3078,3068,3057,3046,3035,3024,
		3013,3002,2990,2979,2968,2957,2946,2934,2923,2912,2900,2889,2877,2866,2854,2843,
		2831,2819,2808,2796,2784,2773,2761,2749,2737,2725,2714,2702,2690,2678,2666,2654,
		2642,2630,2618,2606,2594,2581,2569,2557,2545,2533,2521,2508,2496,2484,2472,2459,
		2447,2435,2422,2410,2398,2385,2373,2360,2348,2335,2323,2311,2298,2286,2273,2261,
		2248,2236,2223,2211,2198,2186,2173,2161,2148,2135,2123,2110,2098,2085,2073,2060,
		2048,2035,2022,2010,1997,1985,1972,1960,1947,1934,1922,1909,1897,1884,1872,1859,
		1847,1834,1822,1809,1797,1784,1772,1760,1747,1735,1722,1710,1697,1685,1673,1660,
		1648,1636,1623,1611,1599,1587,1574,1562,1550,1538,1526,1514,1501,1489,1477,1465,
		1453,1441,1429,1417,1405,1393,1381,1370,1358,1346,1334,1322,1311,1299,1287,1276,
		1264,1252,1241,1229,1218,1206,1195,1183,1172,1161,1149,1138,1127,1116,1105,1093,
		1082,1071,1060,1049,1038,1027,1017,1006,995,984,973,963,952,941,931,920,
		910,900,889,879,869,858,848,838,828,818,808,798,788,778,768,758,
		749,739,729,720,710,701,691,682,672,663,654,645,636,627,618,609,
		600,591,582,573,565,556,547,539,530,522,514,505,497,489,481,473,
		465,457,449,441,433,426,418,410,403,395,388,381,373,366,359,352,
		345,338,331,324,318,311,304,298,291,285,279,272,266,260,254,248,
		242,236,230,224,219,213,207,202,197,191,186,181,176,171,166,161,
		156,151,146,142,137,133,128,124,120,115,111,107,103,99,96,92,
		88,85,81,78,74,71,68,64,61,58,55,53,50,47,44,42,
		39,37,35,32,30,28,26,24,22,20,19,17,15,14,12,11,
		10,9,8,7,6,5,4,3,2,2,1,1,1,0,0,0,
		0,0,0,0,1,1,1,2,2,3,4,5,6,7,8,9,
		10,11,12,14,15,17,19,20,22,24,26,28,30,32,35,37,
		39,42,44,47,50,53,55,58,61,64,68,71,74,78,81,85,
		88,92,96,99,103,107,111,115,120,124,128,133,137,142,146,151,
		156,161,166,171,176,181,186,191,197,202,207,213,219,224,230,236,
		242,248,254,260,266,272,279,285,291,298,304,311,318,324,331,338,
		345,352,359,366,373,381,388,395,403,410,418,426,433,441,449,457,
		465,473,481,489,497,505,514,522,530,539,547,556,565,573,582,591,
		600,609,618,627,636,645,654,663,672,682,691,701,710,720,729,739,
		749,758,768,778,788,798,808,818,828,838,848,858,869,879,889,900,
		910,920,931,941,952,963,973,984,995,1006,1017,1027,1038,1049,1060,1071,
		1082,1093,1105,1116,1127,1138,1149,1161,1172,1183,1195,1206,1218,1229,1241,1252,
		1264,1276,1287,1299,1311,1322,1334,1346,1358,1370,1381,1393,1405,1417,1429,1441,
		1453,1465,1477,1489,1501,1514,1526,1538,1550,1562,1574,1587,1599,1611,1623,1636,
		1648,1660,1673,1685,1697,1710,1722,1735,1747,1760,1772,1784,1797,1809,1822,1834,
		1847,1859,1872,1884,1897,1909,1922,1934,1947,1960,1972,1985,1997,2010,2022,2035};
//uint16_t adc_data_counter = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DAC_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM2_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM2_Init();
  MX_DMA_Init();
  MX_DAC_Init();
  /* USER CODE BEGIN 2 */

//  HAL_ADC_Start_DMA(&hadc1, &dma_buff, 16);
//  HAL_TIM_Base_Start(&htim3);
//  HAL_DAC_Start(&hdac, DAC1_CHANNEL_2)
  HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_2, (uint32_t*)dac_data, DAC_BUFF_LEN, DAC_ALIGN_12B_R);
  HAL_TIM_Base_Start(&htim2);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief DAC Initialization Function
  * @param None
  * @retval None
  */
static void MX_DAC_Init(void)
{

  /* USER CODE BEGIN DAC_Init 0 */

  /* USER CODE END DAC_Init 0 */

  DAC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN DAC_Init 1 */

  /* USER CODE END DAC_Init 1 */
  /** DAC Initialization
  */
  hdac.Instance = DAC;
  if (HAL_DAC_Init(&hdac) != HAL_OK)
  {
    Error_Handler();
  }
  /** DAC channel OUT2 config
  */
  sConfig.DAC_Trigger = DAC_TRIGGER_T2_TRGO;
  sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
  if (HAL_DAC_ConfigChannel(&hdac, &sConfig, DAC_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DAC_Init 2 */

  /* USER CODE END DAC_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 83;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 9;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream6_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream6_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream6_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD4_GPIO_Port, LD4_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : BOOT1_Pin */
  GPIO_InitStruct.Pin = BOOT1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(BOOT1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD4_Pin */
  GPIO_InitStruct.Pin = LD4_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD4_GPIO_Port, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */
//void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc){
//	if(adc_data_counter < ADC_BUFF_LEN){
//	uint16_t ovs = 0;
//	for(uint8_t i = 0; i < DMA_BUFF_LEN; i++){
//		ovs += dma_buff[i];
//	}
//	adc_data[adc_data_counter] = (ovs >> 2); // 12-bit measurements -> 14-bit result
//	adc_data_counter++;
//	HAL_DAC_SetValue(&hdac, DAC1_CHANNEL_2, DAC_ALIGN_12B_L, ovs);
//	}
//	else{
//		HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_12);
//	}
//}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
